import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROOT = "./demo_data_language"
dataset = LeRobotDataset("my_pi0", root=ROOT)

print("=" * 50)
print("Dataset Loaded")
print(f"Total frames: {len(dataset)}")
print(f"Total episodes: {dataset.num_episodes}")
print("=" * 50)

# ---------------------------------------------------
# Hyperparameters
# ---------------------------------------------------

MAX_ACTION = 0.05
MAX_DELTA_STATE = 0.1
MAX_Z = 1.5
MIN_Z = -0.1

bad_frames = []

# ---------------------------------------------------
# Iterate episodes
# ---------------------------------------------------

for ep_idx in range(dataset.num_episodes):

    from_idx = dataset.episode_data_index["from"][ep_idx].item()
    to_idx = dataset.episode_data_index["to"][ep_idx].item()

    print(f"\nChecking Episode {ep_idx}")
    print(f"Frames: {to_idx - from_idx}")

    prev_state = None

    for idx in range(from_idx, to_idx):

        data = dataset[idx]

        state = data["observation.state"].numpy()
        action = data["action"].numpy()

        # ---------------------------------------------------
        # 1. NaN check
        # ---------------------------------------------------

        if np.isnan(state).any():
            print(f"[NaN STATE] frame {idx}")
            bad_frames.append(idx)

        if np.isnan(action).any():
            print(f"[NaN ACTION] frame {idx}")
            bad_frames.append(idx)

        # ---------------------------------------------------
        # 2. Action magnitude check
        # ---------------------------------------------------

        pos_action = action[:3]

        if np.max(np.abs(pos_action)) > MAX_ACTION:
            print(f"[BIG ACTION] frame {idx}")
            print(pos_action)

        # ---------------------------------------------------
        # 3. Gripper check
        # ---------------------------------------------------

        grip = action[-1]

        if grip not in [-1, 1]:
            print(f"[BAD GRIPPER] frame {idx}: {grip}")

        # ---------------------------------------------------
        # 4. EEF z check
        # ---------------------------------------------------

        z = state[2]

        if z > MAX_Z or z < MIN_Z:
            print(f"[BAD Z] frame {idx}: z={z}")

        # ---------------------------------------------------
        # 5. State continuity
        # ---------------------------------------------------

        if prev_state is not None:

            delta = state[:3] - prev_state[:3]

            if np.linalg.norm(delta) > MAX_DELTA_STATE:
                print(f"[STATE JUMP] frame {idx}")
                print(delta)

            # ---------------------------------------------------
            # 6. Replay consistency
            # ---------------------------------------------------

            pred_next = prev_state[:3] + prev_action[:3]

            err = np.linalg.norm(pred_next - state[:3])

            if err > 0.05:
                print(f"[REPLAY MISMATCH] frame {idx}")
                print(f"pred : {pred_next}")
                print(f"real : {state[:3]}")
                print(f"err  : {err:.4f}")

        prev_state = state
        prev_action = action

print("\nFinished checking dataset.")