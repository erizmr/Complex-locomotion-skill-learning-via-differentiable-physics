import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
import glob
import numpy as np

weight_list = []
def load_weights(name="save.pkl"):
    print(name)
    w_val = pkl.load(open(name, 'rb'))
    for val in w_val:
        print(val, val.shape)
        weight_list.append(val)

robot_id = 4
# model_path = glob.glob(f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vhc_5_l05/DiffTaichi_DiffPhy/*/models/weight.pkl")[-1]
# model_path = "./saved_results/sim_config_DiffPhy_robot4_vhc_5_l05/DiffTaichi_DiffPhy/*/models/weight.pkl"
model_path = glob.glob(f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vha/DiffTaichi_DiffPhy/*/models/weight.pkl")[-1]
load_weights(model_path)
weight_1 = weight_list[0]
weight_2 = weight_list[1]

# from IPython import embed
# embed()
#
# exit(0)
print((weight_2 ** 2).mean(axis = 0))
print(abs(weight_2).mean(axis = 0))
sn, sm = weight_1.shape
abs_weight_1 = abs(weight_1)
print(abs_weight_1.shape)
vis_first_k = np.zeros_like(weight_1)
for i in range(sn):
    for j in range(sm):
        for k in range(sm):
            if j != k and abs_weight_1[i, j] < abs_weight_1[i, k]:
                vis_first_k[i, j] += 1.
        vis_first_k[i, j] = pow(0.8, vis_first_k[i, j])

import os
os.makedirs("robot_{}".format(robot_id), exist_ok = True)

def visualize(w, name):
    sns.heatmap(w, cmap='gray')
    plt.title(f"robot_{robot_id}_{name}")
    plt.savefig(f"robot_{robot_id}/{name}.png", dpi = 500)
    plt.clf()

visualize(abs_weight_1, "abs_raw")

for i in range(2, 7):
    visualize(abs_weight_1 > (i / 10.), f"abs_threshold_0.{i}")

visualize(np.floor(abs_weight_1 * 10) / 10 * (abs_weight_1 > 0.2), "abs_floor")

visualize(vis_first_k, "first_k")

rel_weight_1 = abs_weight_1 / abs_weight_1.max(axis = 1)[:, None]
abs_weight_1 = None
visualize(rel_weight_1, "rel_raw")
assert rel_weight_1.max(axis = 1).mean() > 0.999

for i in range(5, 10):
    visualize(rel_weight_1 > (i / 10.), f"rel_threshold_0.{i}")

visualize(np.floor(rel_weight_1 * 10) / 10 * (rel_weight_1 > 0.5), "rel_floor")