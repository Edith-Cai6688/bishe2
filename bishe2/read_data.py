import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt

ROOT = "./demo_data_language5"

dataset = LeRobotDataset("my_pi0", root=ROOT)

print("=" * 50)
print(f"Total episodes: {dataset.num_episodes}")
print(f"Total frames  : {len(dataset)}")
print("=" * 50)

# ---------------------------------------------------
# Choose episode
# ---------------------------------------------------

EPISODE_IDX = 0

from_idx = dataset.episode_data_index["from"][EPISODE_IDX].item()
to_idx = dataset.episode_data_index["to"][EPISODE_IDX].item()

print(f"Episode {EPISODE_IDX}")
print(f"Frames: {to_idx - from_idx}")

# ---------------------------------------------------
# Frame loop
# ---------------------------------------------------

for idx in range(from_idx, to_idx):

    data = dataset[idx]

    # ---------------------------------------------------
    # Read data
    # ---------------------------------------------------

    agent_img = data["observation.image"].numpy()
    wrist_img = data["observation.wrist_image"].numpy()

    state = data["observation.state"].numpy()
    action = data["action"].numpy()

    task = data["task"]

    # ---------------------------------------------------
    # Convert image
    # Lerobot image format:
    # (C,H,W) -> (H,W,C)
    # ---------------------------------------------------

    agent_img = np.transpose(agent_img, (1, 2, 0))
    wrist_img = np.transpose(wrist_img, (1, 2, 0))

    # float -> uint8
    agent_img = (agent_img * 255).astype(np.uint8)
    wrist_img = (wrist_img * 255).astype(np.uint8)

    # ---------------------------------------------------
    # Print info
    # ---------------------------------------------------

    print("\n" + "=" * 30)
    print(f"Frame : {idx}")
    print(f"Task  : {task}")

    print(f"\nState:")
    print(state)

    print(f"\nAction:")
    print(action)

    # ---------------------------------------------------
    # Concatenate images
    # ---------------------------------------------------

    vis = np.concatenate([agent_img, wrist_img], axis=1)

    # BGR for cv2
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    plt.imshow(vis)
    plt.title(f"Frame {idx}")
    plt.axis("off")
    plt.show()


cv2.destroyAllWindows()