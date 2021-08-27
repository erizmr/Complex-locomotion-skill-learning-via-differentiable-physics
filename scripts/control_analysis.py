import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

task_name = "long"
all_robots = sorted(glob.glob("./stats/*"))
fig, axes = plt.subplots(2, 2, figsize=(25, 16))
axes = axes.flatten()
for i, robot_path in enumerate(all_robots):
    # robot_path = all_robots[0]
    robot_id = robot_path.split("/")[-1].split('_')[-1]
    all_controls = sorted(glob.glob(os.path.join(robot_path, '*')))
    data_dict = {}
    all_df = []
    target_v = None
    for c in all_controls:
        control_length = c.split('_l')[-1]
        df = pd.read_csv(os.path.join(c, f"stats_{task_name}.csv"))
        all_df.append(df)
        data_dict[control_length] = df["v"]
        if target_v is None:
            target_v = df["target_v"]

    axes[i].plot([x for x in range(len(target_v))], target_v, label="target v", color='tab:blue', alpha=0.5)
    for k, v in data_dict.items():
        axes[i].plot([x for x in range(len(target_v))], v, label="Control length "+k)

    axes[i].set_title(f"Robot ID {robot_id}")
    axes[i].set_ylabel("Velocity")
    axes[i].set_xlabel("Steps")
    axes[i].legend()

fig.suptitle(f'Target velocity tracking ', fontsize=20)
    # axes[i].show()

plt.show()