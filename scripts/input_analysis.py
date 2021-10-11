import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
import glob
import numpy as np
robot_names = {2:"Alpaca", 3:"Monster", 4:"HugeStool", 5:"Stool", 6:"Snake"}
weight_list = []
def load_weights(name="save.pkl"):
    print(name)
    w_val = pkl.load(open(name, 'rb'))
    for val in w_val:
        print(val, val.shape)
        weight_list.append(val)

robot_id = 2
# model_path = glob.glob(f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vhc_5_l05/DiffTaichi_DiffPhy/*/models/weight.pkl")[-1]
# model_path = "./saved_results/sim_config_DiffPhy_robot4_vhc_5_l05/DiffTaichi_DiffPhy/*/models/weight.pkl"
# model_path = glob.glob(f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vha/DiffTaichi_DiffPhy/*/models/weight.pkl")[-1]
model_path = glob.glob(f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vh/DiffTaichi_DiffPhy/*/models/weight.pkl")[-1]
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
os.makedirs("imgs", exist_ok = True)

def visualize(w, name):
    if name == "abs_raw":
        name = "Weights Absolute Values"
    if name == "first_k":
        name = "Top N Values of Rows"
    ax = sns.heatmap(w, square=True, cbar=True, linewidths=.01, cmap="YlGnBu", cbar_kws={"shrink": .48})
    # ax.set_title(f"{robot_names[robot_id]} (A) - {name}", fontsize=32)
    ax.set_title(f"{robot_names[robot_id]} - {name}", fontsize=11)
    ax.set_xlabel("Input Layer Channel", fontsize=10)
    ax.set_ylabel("Hidden Layer Channel", fontsize=10)
    ax.set_yticks(np.array(range(0, 64, 9)) + 0.5)
    ax.set_yticklabels(range(0, 64, 9), fontsize=9)
    #ax.set_xticklabels(range(0, 202, 8), fontsize=9)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=10)
    plt.savefig(f"imgs/robot_{robot_id}_weights_{name}.pdf", dpi = 1000, bbox_inches='tight')
    # plt.show()
    plt.clf()
    # exit(0)

visualize(abs_weight_1, "abs_raw")

# for i in range(2, 7):
#     visualize(abs_weight_1 > (i / 10.), f"abs_threshold_0.{i}")
#
# visualize(np.floor(abs_weight_1 * 10) / 10 * (abs_weight_1 > 0.2), "abs_floor")

visualize(vis_first_k, "first_k")

# rel_weight_1 = abs_weight_1 / abs_weight_1.max(axis = 1)[:, None]
# abs_weight_1 = None
# visualize(rel_weight_1, "rel_raw")
# assert rel_weight_1.max(axis = 1).mean() > 0.999
#
# for i in range(5, 10):
#     visualize(rel_weight_1 > (i / 10.), f"rel_threshold_0.{i}")
#
# visualize(np.floor(rel_weight_1 * 10) / 10 * (rel_weight_1 > 0.5), "rel_floor")