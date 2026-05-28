# %% [markdown]
# ### [Optional] Download collected dataset

# %%
# '''
# If you want to use the collected dataset, please download it from Hugging Face.
# '''
# !git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language

# %% [markdown]
# # Visualize your data
# 
# <img src="./media/data_v2.gif" width="480" height="360">
# 
# Visualize your action based on the reconstructed simulation scene. 
# 
# The main simulation is replaying the action.
# 
# The overlayed images on the top right and bottom right are from the dataset. 

# %%
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
ROOT = "./demo_data_language" # The root directory to save the demonstrations 
# If you have downloaded the dataset from Hugging Face, you can set the root to the directory where the dataset is stored
# ROOT = './omy_pnp_language' # if you want to use the example data provided, root = './omy_pnp_language' instead!
dataset = LeRobotDataset('my_pi0', root=ROOT) # if youu want to use the example data provided, root = './omy_pnp_language' instead!

# If you want to use the collected dataset, please download it from Hugging Face.
# dataset = LeRobotDataset('omy_pnp_language', root='omy_pnp_language')

# %% [markdown]
# ## Load Dataset

# %%
import torch

class EpisodeSampler(torch.utils.data.Sampler):
    """
    Sampler for a single episode
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

# %%
# Select an episode index that you want to visualize
# episode_index = 0

# episode_sampler = EpisodeSampler(dataset, episode_index)
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     num_workers=1,
#     batch_size=1,
#     sampler=episode_sampler,
# )


# %% [markdown]
# ## Visualize your Dataset on Simulation

# %%
from env.MyEnv3 import MyEnv
xml_path = './asset/example_scene.xml'
myenv = MyEnv(xml_path, action_type='eef_pose')

for ep_idx in range(200):
    if not myenv.env.is_viewer_alive():
        break
        
    print(f"Currently playing episode: {ep_idx}")
    
    # 为当前 Episode 创建采样器和加载器
    episode_sampler = EpisodeSampler(dataset, ep_idx)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        sampler=episode_sampler,
    )
    
    iter_dataloader = iter(dataloader)
    myenv.reset()
    step = 0
    

    # 内层循环：播放当前 Episode 的每一帧
    while step < len(episode_sampler):
        if not myenv.env.is_viewer_alive():
            break
            
        myenv.step_env()
        
        # 维持仿真频率（例如 20Hz）
        if myenv.env.loop_every(HZ=20):
            try:
                data = next(iter_dataloader)
            except StopIteration:
                break

            if step == 0:
                # 每一局开始时，初始化物体位置和指令
                instruction = data['task'][0]
                print(f"Instruction: {instruction}")
                myenv.set_instruction(instruction)
                myenv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0, 3:6], data['obj_init'][0, 6:9])
                q_init = data['robot_init'][0].numpy()

                myenv.env.forward(
                    q=q_init,
                    joint_names=myenv.joint_names,
                    increase_tick=False
                )
            # 获取动作并执行
            action = data['action'].numpy()
            print(action[0][:3])
            obs = myenv.step(action[0])

            # 同步显示数据集中的图像（右上角/右下角叠加）
            myenv.rgb_agent = (data['observation.image'][0].numpy() * 255).astype(np.uint8)
            myenv.rgb_ego = (data['observation.wrist_image'][0].numpy() * 255).astype(np.uint8)
            
            # 维度转换 (C, H, W) -> (H, W, C)
            myenv.rgb_agent = np.transpose(myenv.rgb_agent, (1, 2, 0))
            myenv.rgb_ego = np.transpose(myenv.rgb_ego, (1, 2, 0))
            
            myenv.render()
            step += 1

print("Visualization finished.")
myenv.env.close_viewer()
# %%
# step = 0
# iter_dataloader = iter(dataloader)
# myenv.reset()

# while myenv.env.is_viewer_alive():
#     myenv.step_env()
#     if myenv.env.loop_every(HZ=20):
#         # Get the action from dataset
#         data = next(iter_dataloader)
#         if step == 0:
#             # Reset the object pose based on the dataset
#             instruction = data['task'][0]
#             print(f"instruction is {instruction}")
#             myenv.set_instruction(instruction)
#             myenv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0, 3:6], data['obj_init'][0, 6:9])
#         # Get the action from dataset
#         action = data['action'].numpy()
#         # print(f"action is {action[0]}")
#         obs = myenv.step(action[0])

#         # Visualize the image from dataset to rgb_overlay
#         myenv.rgb_agent = data['observation.image'][0].numpy()*255
#         myenv.rgb_ego = data['observation.wrist_image'][0].numpy()*255
#         myenv.rgb_agent = myenv.rgb_agent.astype(np.uint8)
#         myenv.rgb_ego = myenv.rgb_ego.astype(np.uint8)
#         # 3 256 256 -> 256 256 3
#         myenv.rgb_agent = np.transpose(myenv.rgb_agent, (1,2,0))
#         myenv.rgb_ego = np.transpose(myenv.rgb_ego, (1,2,0))
#         myenv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
#         myenv.render()
#         step += 1

#         if step == len(episode_sampler):
#             # start from the beginning
#             iter_dataloader = iter(dataloader)
#             myenv.reset()
#             step = 0
#     # myenv

# # %%
# myenv.env.close_viewer()

# %%
dataset.push_to_hub(upload_large_folder=True)

# %%



