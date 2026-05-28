import sys
import random
import numpy as np
import os
from PIL import Image
from env.MyEnv3 import MyEnv
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import glfw


# If you want to randomize the object positions, set this to None
# If you fix the seed, the object positions will be the same every time
SEED = None 
# SEED = None <- Uncomment this line to randomize the object positions

REPO_NAME = 'my_pi0'
NUM_DEMO = 300 # Number of demonstrations to collect
ROOT = "./demo_data_language" # The root directory to save the demonstrations

xml_path = './asset/example_scene.xml'
# Define the environment
myenv = MyEnv(xml_path, seed = SEED, state_type = 'joint_angle')

## Define Dataset Fatures and Create your dataset!
# The dataset is contained as follows:
# ```
# fps = 20,
# features={
#     "observation.image": {
#         "dtype": "image",
#         "shape": (256, 256, 3),
#         "names": ["height", "width", "channels"],
#     },
#     "observation.wrist_image": {
#         "dtype": "image",
#         "shape": (256, 256, 3),
#         "names": ["height", "width", "channel"],
#     },
#     "observation.state": {
#         "dtype": "float32",
#         "shape": (6,),
#         "names": ["state"], # x, y, z, roll, pitch, yaw
#     },
#     "action": {
#         "dtype": "float32",
#         "shape": (7,),
#         "names": ["action"], # 6 joint angles and 1 gripper
#     },
#     "obj_init": {
#         "dtype": "float32",
#         "shape": (6,),
#         "names": ["obj_init"], # just the initial position of the object. Not used in training.
#     },
# },
# ```


# This will make the dataset on './demo_data' folder, which will look like this,

# ```
# .
# ├── data
# │   ├── chunk-000
# │   │   ├── episode_000000.parquet
# │   │   └── ...
# ├── meta
# │   ├── episodes.jsonl
# │   ├── info.json
# │   ├── stats.json
# │   └── tasks.jsonl
# └── 
# ```


create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
    else:
        create_new = False


if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT, 
                robot_type="omy",
                fps=20, # 20 frames per second
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (8,),
                        "names": ["state"], # 7 joint angles
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # dx dy dz dr dp dyaw gripper
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (9,),
                        "names": ["obj_init"], # just the initial position of the object. Not used in training.
                    },
                    "robot_init": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["robot_init"],
                    }
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)


episode_id = 0
max_dist = 0.035  
stage = 0
grasp_counter = 0
release_counter = 0
action = np.zeros(7)
MAX_STEPS_PER_STAGE = 300  # 每个stage最大步数
MAX_TOTAL_STEPS = 2000      # 整个episode最大步数
stage_steps = 0
total_steps = 0
failed = False

# Get obj position
red_cube_pose, green_cube_pose, plate_pose = myenv.get_obj_pose()
cube_pose = myenv.get_target_pose()
while myenv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    myenv.step_env()
    if myenv.env.loop_every(HZ=20):
        agent_image, wrist_image = myenv.grab_image()
        current_joint_state = myenv.get_joint_state()[:7]

        # Get eef postion and rotaion
        eef_pose = myenv.get_ee_pose()
        target_rpy = np.array([np.pi, 0, np.pi/2])

        stage_steps += 1
        total_steps += 1
        
        # Check failture
        # 1. Run out of time
        if total_steps > MAX_TOTAL_STEPS:
            print(f"❌ Episode {episode_id} failed: Total steps timeout ({MAX_TOTAL_STEPS} steps)")
            failed = True
        
        # 2. Stage failtire
        if stage_steps > MAX_STEPS_PER_STAGE:
            print(f"❌ Episode {episode_id} failed: Stage {stage} stuck for {MAX_STEPS_PER_STAGE} steps")
            failed = True
        
        # 3. Robot lose control
        if eef_pose[2] < -0.1 or eef_pose[2] > 1.5:  # z轴超出合理范围
            print(f"❌ Episode {episode_id} failed: EEF out of range (z={eef_pose[2]:.2f})")
            failed = True
        
        # Something wrong happened, reset the env
        if failed:
            print(f"Resetting episode {episode_id}...")
            myenv.reset()
            dataset.clear_episode_buffer()  # 清空当前episode数据
            # Reset all the variables
            stage = 0
            grasp_counter = 0
            release_counter = 0
            stage_steps = 0
            total_steps = 0
            failed = False
            action = np.zeros(7)
            # 重新获取物体位置
            red_cube_pose, green_cube_pose, plate_pose = myenv.get_obj_pose()
            cube_pose = myenv.get_target_pose()
            continue  # 跳过当前帧，重新开始

        # Check if the episode is done
        done = myenv.check_success()
        if done:
            # Save the episode data and reset the environment
            print(f"Success recording No. {episode_id} data!")
            dataset.save_episode()
            myenv.reset()
            stage = 0
            grasp_counter = 0
            release_counter = 0
            stage_steps = 0
            total_steps = 0
            failed = False
            action = np.zeros(7)
            episode_id += 1
            red_cube_pose, green_cube_pose, plate_pose = myenv.get_obj_pose()
            cube_pose = myenv.get_target_pose()
            continue
        
        # Move to the cube
        if stage == 0:
            target_pose = cube_pose + np.array([0.01, 0, 0.3])
            diff_pos = target_pose - eef_pose[:3]
            action[:3] = np.clip(diff_pos, -0.01, 0.01)
            action[:3] += np.random.normal(0, 0.002, size=3)
            # Incremental
            # diff_pos = (target_pose - eef_pose[:3]) 
            # diff_rpy = (target_rpy - eef_pose[3:6]) 
            # action[:3] = np.clip(diff_pos, -0.01, 0.01) # 每帧最多动 1cm
            # action[3:6] = np.clip(diff_rpy, -0.05, 0.05)
            # action[-1] = -1 
            # Absolute
            # action[:3] = target_pose
            # action[3:6] = target_rpy
            delta_rpy = target_rpy - eef_pose[3:6]
            action[3:6] = 0
            action[-1] = -1
            print(f"The target pos is {target_pose}")
            print(f"The eef pos is {eef_pose[:3]}")
            print(np.linalg.norm(target_pose - eef_pose[:3]))
            if np.linalg.norm(target_pose - eef_pose[:3]) < 0.02:
                stage = 1
                stage_steps = 0
        # Go down
        elif stage == 1:
            target_pose = cube_pose + np.array([0.01, 0, 0.01])
            # action[:3] = target_pose
            diff_pos = target_pose - eef_pose[:3]
            action[:3] = np.clip(diff_pos, -0.01, 0.01)
            action[:3] += np.random.normal(0, 0.002, size=3)
            print(f"stage2: action is {action[:3]}")
            delta_rpy = target_rpy - eef_pose[3:6]
            action[3:6] = 0
            action[-1] = -1
            if np.linalg.norm(target_pose - eef_pose[:3]) < 0.02:
                stage = 2
                stage_steps = 0
            # print(f"The cube pos is {cube_pose}")
            # print(f"The target pos is {target_pose}")
            # print(f"The eef pos is {eef_pose[:3]}")
            
        # Close the gripper
        elif stage == 2:
            target_pose = cube_pose + np.array([0.01, 0, 0.01])
            diff_pos = target_pose - eef_pose[:3]
            action[:3] = np.clip(diff_pos, -0.01, 0.01)
            action[3:6] = 0
            action[-1] = 1
            grasp_counter += 1
            if grasp_counter > 10:
                stage = 3
                stage_steps = 0
                grasp_counter = 0
                myenv.cube_pose = myenv.get_target_pose()
            
        # Lift up
        elif stage == 3:
            target_pose = myenv.cube_pose + np.array([0, 0, 0.2])
            diff_pos = target_pose - eef_pose[:3]
            # print(f"target pos is  {target_pose}")
            # print(f"eef pos is {eef_pose[:3]}")
            print(f"cube is {cube_pose}")
            action[:3] = np.clip(diff_pos, -0.02, 0.02)
            # action[:3] += np.random.normal(0, 0.002, size=3)
            # print(f"action is {action[:3]}")
            action[3:6] = 0
            action[-1] = 1
            if np.linalg.norm(target_pose - eef_pose[:3]) < 0.02:
                stage = 4
                stage_steps = 0
            
        # Move to the plate
        elif stage == 4:
            target_pose = plate_pose + np.array([0.01, 0, 0.2])
            # action[:3] = target_pose
            diff_pos = target_pose - eef_pose[:3]
            action[:3] = np.clip(diff_pos, -0.02, 0.02)
            action[:3] += np.random.normal(0, 0.002, size=3)
            delta_rpy = target_rpy - eef_pose[3:6]
            action[3:6] = 0
            action[-1] = 1
            if np.linalg.norm(target_pose - eef_pose[:3]) < 0.02:
                stage = 5
                stage_steps = 0
            # print(f"The plate pos is {plate_pose}")
            # print(f"The target pos is {target_pose}")
            # print(f"The eef pos is {eef_pose[:3]}")
            
        # Go down
        elif stage == 5:
            target_pose = plate_pose + np.array([0.01, 0, 0.04])
            # action[:3] = target_pose
            diff_pos = target_pose - eef_pose[:3]
            action[:3] = np.clip(diff_pos, -0.02, 0.02)
            action[:3] += np.random.normal(0, 0.002, size=3)
            delta_rpy = target_rpy - eef_pose[3:6]
            action[3:6] = 0
            action[-1] = 1
            if np.linalg.norm(target_pose - eef_pose[:3]) < 0.02:
                stage = 6
                stage_steps = 0
            
        # Open the gripper
        elif stage == 6:
            target_pose = plate_pose + np.array([0.01, 0, 0.04])
            diff_pos = target_pose - eef_pose[:3]
            action[:3] = np.clip(diff_pos, -0.02, 0.02)
            action[3:6] = 0
            action[-1] = -1
            release_counter += 1
            if release_counter > 20:
                stage = 7
                stage_steps = 0
                release_counter = 0
                
        # Go up
        elif stage == 7:
            target_pose = plate_pose + np.array([0.01, 0, 0.3])
            # action[:3] = target_pose
            diff_pos = target_pose - eef_pose[:3]
            action[:3] = np.clip(diff_pos, -0.03, 0.03)
            action[:3] += np.random.normal(0, 0.002, size=3)
            action[3:6] = 0
            action[-1] = -1

        # # Get the end-effector pose and images
        # agent_image, wrist_image = myenv.grab_image()
        # resize to 256x256
        agent_image = Image.fromarray(agent_image)
        wrist_image = Image.fromarray(wrist_image)
        agent_image = agent_image.resize((256, 256))
        wrist_image = wrist_image.resize((256, 256))
        agent_image = np.array(agent_image)
        wrist_image = np.array(wrist_image)
        # Compute the target joint angles from eef target pose(7) without updating positions
        current_eef = myenv.step(action)
        # Get target joint angles 
        final_action = action.astype(np.float32)
        dataset.add_frame({
            "observation.image": agent_image,
            "observation.wrist_image": wrist_image,
            "observation.state": current_eef, # Without gripper
            "action": final_action, # With gripper
            "obj_init": myenv.obj_init_pose,
            "robot_init": myenv.robot_init_q.astype(np.float32)
            }, task = myenv.instruction
        )

        myenv.render(idx = episode_id)

myenv.env.close_viewer()

# import shutil
# shutil.rmtree(dataset.root / "images")
    