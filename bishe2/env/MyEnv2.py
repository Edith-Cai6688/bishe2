import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from env.mujoco_parser import MuJoCoParserClass
from env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from env.ik import solve_ik
from env.transforms import rpy2r, r2rpy
import os
import copy
import glfw
import time

class MyEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                seed = None):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Render',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names = ['robot0_joint1',
                    'robot0_joint2',
                    'robot0_joint3',
                    'robot0_joint4',
                    'robot0_joint5',
                    'robot0_joint6',
                    'robot0_joint7']
        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed != None: np.random.seed(seed) 
        q_init = np.deg2rad([0,0,0,0,0,0,0]) #机器人的所有7个关节都处于零位（直立状态）
        # 求解七个关节角度
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'gripper0_right_eef',
            q_init       = q_init, # ik from zero pose
            p_trgt       = np.array([0.3,0.3,0.8]),
            R_trgt       = rpy2r(np.deg2rad([180, 0, 90])),
        )
        # 不考虑实际物理，直接将机械臂执行7个关节角度变化
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)
        # p, R = self.env.get_pR_body(body_name='gripper0_right_eef')
        # print(f"Z轴方向: {R[:, 2]}")
        # print(f"与期望[0,0,-1]的夹角: {np.arccos(-R[2,2]) * 180 / np.pi:.2f}度")
        # print(f"IK 求解后的关节角度 (rad): \n{q_zero}")
        # start_time = time.time()
        # while time.time() - start_time < 50:
        #     self.env.render()
        
        # set plate position
        plate_xyz = np.array([0.6, -0.25, 0.0])
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=plate_xyz)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        # Set object positions
        both_xyzs = sample_xyzs(
        2,  # 一次生成两个位置
        x_range   = [+0.4, +0.7],   # 扩大范围，让红绿可以出现在任何地方
        y_range   = [-0.15, +0.25], # 扩大y范围
        z_range   = [0.0, 0.0],
        min_dist  = 0.12,           # 两个方块之间的最小距离
        xy_margin = 0.05
    )
    
        # 随机分配哪个位置给红色，哪个给绿色
        if np.random.random() > 0.5:
            self.env.set_p_base_body(body_name='cube_main', p=both_xyzs[0, :])      # 红色
            self.env.set_p_base_body(body_name='cube_main_2', p=both_xyzs[1, :])    # 绿色
        else:
            self.env.set_p_base_body(body_name='cube_main', p=both_xyzs[1, :])      # 红色
            self.env.set_p_base_body(body_name='cube_main_2', p=both_xyzs[0, :])    # 绿色

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([0.0]*2)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='gripper0_right_eef')
        redcube_init_pose, greencube_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([redcube_init_pose, greencube_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        self.set_instruction()
        print("DONE INITIALIZATION")
        self.gripper_state = True
        self.past_chars = []

    def set_instruction(self, given = None):
        """
        Set the instruction for the task
        """
        if given is None:
            if not hasattr(self, '_instruction_counter'):
                self._instruction_counter = 0
        
            # 交替选择红色和绿色
            if self._instruction_counter % 2 == 0:
                obj1 = 'red'
            else:
                obj1 = 'green'
            
            self._instruction_counter += 1
            
            self.instruction = f'Place the {obj1} cube on the plate.'
            if obj1 == 'red':
                self.obj_target = 'cube_main'
            else:
                self.obj_target = 'cube_main_2'
        else:
            self.instruction = given
            if 'red' in self.instruction:
                self.obj_target = 'cube_main'
            elif 'green' in self.instruction:
                self.obj_target = 'cube_main_2'
            else:
                raise ValueError('Instruction does not contain a valid object color (red or green).')

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'gripper0_right_eef',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        gripper_cmd = np.zeros(2)
        if action[-1] > 0.5:  # 张开
            gripper_cmd[0] = 0.04   
            gripper_cmd[1] = -0.04  
        else:  # 闭合
            gripper_cmd[0] = 0.0    
            gripper_cmd[1] = 0.0    
        # gripper_cmd[[1,3]] *= 0.8
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')
        
    # def step(self, action):
    #     '''
    #     Take a step in the environment
    #     args:
    #         action: np.array of shape (9,), action to take
    #     returns:
    #         state: np.array, state of the environment after taking the action
    #             - ee_pose: [px,py,pz,r,p,y]
    #             - joint_angle: [j1,j2,j3,j4,j5,j6]

    #     '''
    #     if self.action_type == 'eef_pose':
    #         # 1. Get current eef pose
    #         q = self.env.get_qpos_joints(joint_names=self.joint_names)
    #         # Incremental
    #         # current_p, current_R = self.env.get_pR_body(body_name='gripper0_right_eef')
    #         # target_p = current_p + action[:3]
    #         # target_R = current_R.dot(rpy2r(action[3:6]))
    #         # Absolute
    #         target_p = action[:3]
    #         target_R = rpy2r(action[3:6])
    #         q ,ik_err_stack,ik_info = solve_ik(
    #             env                = self.env,
    #             joint_names_for_ik = self.joint_names,
    #             body_name_trgt     = 'gripper0_right_eef',
    #             q_init             = q,
    #             p_trgt             = target_p,
    #             R_trgt             = target_R,
    #             max_ik_tick        = 100,
    #             ik_stepsize        = 0.3,
    #             ik_eps             = 1e-2,
    #             ik_th              = np.radians(5.0),
    #             render             = False,
    #             verbose_warning    = True,
    #         )

    #     elif self.action_type == 'delta_joint_angle':
    #         q = action[:-1] + self.last_q
    #     elif self.action_type == 'joint_angle':
    #         q = action[:-1]
    #     else:
    #         raise ValueError('action_type not recognized')
        
    #     gripper_cmd = np.zeros(2)
    #     if action[-1] < 0:  # 张开
    #         gripper_cmd[0] = 0.04   
    #         gripper_cmd[1] = -0.04  
    #     else:  # 闭合
    #         gripper_cmd[0] = 0.0    
    #         gripper_cmd[1] = 0.0    
    #     # gripper_cmd[[1,3]] *= 0.8
    #     self.compute_q = q
    #     q = np.concatenate([q, gripper_cmd])

    #     self.q = q
    #     if self.state_type == 'joint_angle':
    #         return self.get_joint_state()
    #     elif self.state_type == 'ee_pose':
    #         return self.get_ee_pose()
    #     elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
    #         dq =  self.get_delta_q()
    #         return dq
    #     else:
    #         raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='robotview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='robot0_eye_in_hand')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    # def render(self, teleop=False, idx = 0):
    #     '''
    #     Render the environment
    #     '''
    #     self.env.plot_time()
    #     p_current, R_current = self.env.get_pR_body(body_name='gripper0_right_eef')
    #     R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
    #     # self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
    #     # self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
    #     rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
    #     rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
    #     self.env.plot_T(p = np.array([0.1,0.0,1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False)
    #     self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
    #     self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
    #     if teleop:
    #         rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
    #         self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
    #         self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
    #         self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
    #     if getattr(self, 'instruction', None) is not None:
    #         language_instructions = self.instruction
    #         self.env.viewer_text_overlay(text1='Language Instructions',text2=language_instructions)
    #     self.env.render()

    def render(self, teleop=False, idx = 0):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='gripper0_right_eef')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        # self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        # self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        self.env.plot_T(p = np.array([0.1,0.0,1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False)
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        if getattr(self, 'instruction', None) is not None:
            language_instructions = self.instruction
            self.env.viewer_text_overlay(text1='Language Instructions',text2=language_instructions)
                    
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,j7,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('gripper0_right_finger_joint1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('gripper0_right_finger_joint1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        ['body_obj_mug_5', 'body_obj_plate_11']
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_cube = self.env.get_p_body(self.obj_target)
        p_plate = self.env.get_p_body('body_obj_plate_11')
        gripper_pos = self.env.get_qpos_joint('gripper0_right_finger_joint1')
        is_gripper_open = gripper_pos[0] > 0.02  # 阈值根据实际调节
        if np.linalg.norm(p_cube[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_cube[2] - p_plate[2]) < 0.05 and is_gripper_open:
            p = self.env.get_p_body('gripper0_right_eef')[2]
            if p > 0.5:
                return True
        return False
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        p_cube_red = self.env.get_p_body('cube_main')
        p_cube_green = self.env.get_p_body('cube_main_2')

        p_plate = self.env.get_p_body('body_obj_plate_11')

        return p_cube_red, p_cube_green, p_plate
    
    def get_target_pose(self):
        '''
        returns: 
            p_cube: np.array, position of the target pose
        '''
        p_cube = self.env.get_p_body(self.obj_target)

        return p_cube
    
    def set_obj_pose(self, p_cubered, p_cubegreen, p_plate):
        '''
        Set the object poses
        args:
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='cube_main',p=p_cubered)
        self.env.set_R_base_body(body_name='cube_main',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='cube_main_2',p=p_cubegreen)
        self.env.set_R_base_body(body_name='cube_main_2',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='gripper0_right_eef')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)