import glfw
from env.Myenv_place import MyEnv
import mujoco
import numpy as np

xml_path = './asset/example_scene.xml'

myenv = MyEnv(xml_path)

last_p_state = False
episode_id = 0

while myenv.env.is_viewer_alive():

    myenv.step_env()

    # 必须先 grab image
    myenv.grab_image()

    # render
    myenv.render(idx=episode_id)

    # 检测 P 键
    current_p_state = glfw.get_key(
        myenv.env.viewer.window,
        glfw.KEY_P
    ) == glfw.PRESS

    # 只打印一次
    if current_p_state and not last_p_state:

        cam = myenv.env.viewer.cam

        # 获取 camera 参数
        lookat = np.array(cam.lookat)
        distance = cam.distance
        azimuth = np.deg2rad(cam.azimuth)
        elevation = np.deg2rad(cam.elevation)

        # 计算相机位置
        x = lookat[0] + distance * np.cos(elevation) * np.sin(azimuth)
        y = lookat[1] - distance * np.cos(elevation) * np.cos(azimuth)
        z = lookat[2] + distance * np.sin(-elevation)

        pos = np.array([x, y, z])

        # 构造 forward direction
        forward = lookat - pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0, 0, 1])

        right = np.cross(up, forward)
        right /= np.linalg.norm(right)

        up = np.cross(forward, right)

        R = np.stack([right, up, forward], axis=1)

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())

        print("\n========== FIXED CAMERA XML ==========\n")

        print(f'''
    <camera
        mode="fixed"
        name="robotview"
        pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}"
        quat="{quat[0]:.3f} {quat[1]:.3f} {quat[2]:.3f} {quat[3]:.3f}"/>
    ''')

        print("=====================================\n")

    last_p_state = current_p_state

myenv.env.close_viewer()