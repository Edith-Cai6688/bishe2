import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "my_pi0",
    root="./demo_data_language5_100"
)

grippers = []

for i in range(len(dataset)):
    action = dataset[i]["action"].numpy()
    grippers.append(action[-1])

grippers = np.array(grippers)

print("total frames:", len(grippers))
print("open  (-1):", np.sum(grippers < 0))
print("close (+1):", np.sum(grippers > 0))

print("open ratio :", np.mean(grippers < 0))
print("close ratio:", np.mean(grippers > 0))