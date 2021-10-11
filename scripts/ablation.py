import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.ndimage.filters import gaussian_filter1d
from collections import OrderedDict, defaultdict

robot_names = {2:"Alpaca", 3:"Monster", 4:"HugeStool", 5:"Stool"}
h_num = 1
w_num = 4
fig, axes = plt.subplots(h_num, w_num, figsize=(w_num * 7, h_num * 6))
axes = axes.flatten()

# Reference
def get_all_robot(whole_path):
    all_ours_folders = glob.glob(whole_path)
    ours_path_dict = {}
    # print(all_ours_folders)
    for path in all_ours_folders:
        #print(path)
        bs = int(path.split('robot')[-1].split('_')[0])
        #print(bs)
        ours_path_dict[bs] = glob.glob(os.path.join(path, "*"))
        #print(ours_path_dict)
    ours_data_dict = {}
    for k, ps in ours_path_dict.items():
        ours_data_dict[k] = []

        for p in ps:
            ret = pd.read_csv(os.path.join(p, "validation/summary.csv"))
            ret['Unnamed: 0'] = [name.split('_')[0] for name in ret['Unnamed: 0']]
            ret.set_index('Unnamed: 0', inplace=True)
            # print(ret)
            ret = ret.groupby(['Unnamed: 0']).mean()
            ours_data_dict[k].append(ret)
    return ours_data_dict

path_list = [
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01/DiffTaichi_DiffPhy",
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01_hu2019/DiffTaichi_DiffPhy",
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l10_batch_1/DiffTaichi_DiffPhy",
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01_sgd/DiffTaichi_DiffPhy",
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l10_tanh/DiffTaichi_DiffPhy",
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01_no_target/DiffTaichi_DiffPhy",
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01_no_state_vector/DiffTaichi_DiffPhy",
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01_no_periodic/DiffTaichi_DiffPhy",
"./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01_naive_loss/DiffTaichi_DiffPhy"
]

label_list = "Full,Difftaichi,Full-BS,Full-OP,Full-AF,Full-TG,Full-SV,Full-PS,Full-LD".split(",")
data_dicts = [get_all_robot(path) for path in path_list]

last_values_collector = defaultdict(list)
std_collector = defaultdict(list)
normalize = True
save = True
task = "height"
ylabel_name = task
for k in data_dicts[0].keys():
    if k == 55:
        continue
    def draw_single_robot(index, data_dict, label, task = "task"):
        if index not in data_dict.keys():
            print(f"WARNING robot {index} index {label} not found")
            return [0.0], [0.0]
        if (task == "height" or task == "crawl") and label == "Full-LD":
            return [0.0], [0.0]
        vals = data_dict[index]
        X = [int(x) for x in vals[0].columns]
        if task == "task":
            # Remove crawl
            combined = np.array([v.loc[task] - v.loc["crawl"] for v in vals])
        else:
            combined = np.array([v.loc[task] for v in vals])
        mean = combined.mean(axis=0)
        std = combined.std(axis=0)
        base_loss = max(mean)
        ysmoothed = gaussian_filter1d(mean / base_loss, sigma=0.8)
        axes[index-2].plot(X, ysmoothed, label=label)
        axes[index-2].fill_between(X, ysmoothed - std, ysmoothed + std, alpha=0.2)
        return mean / base_loss, std

    for id, data_dict in enumerate(data_dicts):
        loss, std = draw_single_robot(k, data_dict, label_list[id], task)
        last_values_collector[k].append((label_list[id], loss[-1]))
        std_collector[k].append((label_list[id], std[-1]))

    axes[k-2].set_xlabel("Iteartions", fontsize=20)
    if task == "task":
        ylabel_name = "Task"
    if task == "velocity":
        ylabel_name = "Run."
    if task == "height":
        ylabel_name = "Jump."
    if task == "crawl":
        ylabel_name = "Crawl."
    axes[k-2].set_ylabel(f"Normlized Validation {ylabel_name} Loss", fontsize=20)
    axes[k-2].set_title(f"Ablation Study on {robot_names[int(k)]}", fontsize=20)
    axes[k-2].tick_params(axis='x', labelsize=16)
    axes[k-2].tick_params(axis='y', labelsize=16)
    axes[k-2].legend(fontsize=14)


plt.tight_layout()

# fig = plt.gcf()
# fig.legend(
# labels_dict.values(),
#     labels_dict.keys(),
#     loc=(0.0, 0.0),
#     ncol=4,
#     mode="expand",
#     handlelength=2.0,
#     columnspacing=3.0,
#     edgecolor='white',
#     framealpha=0.,
#     fontsize=14
# )

fig = plt.gcf()
img_save_name = f"imgs/all_ablation_{task}"
if save:
    with PdfPages(img_save_name + ".pdf") as pdf:
        pdf.savefig(fig)

plt.show()

print("alpaca ", task, last_values_collector[2], std_collector[2])
print("monster ", task, last_values_collector[3], std_collector[3])
print("hugestool ", task, last_values_collector[4], std_collector[4])
print("stool", task, last_values_collector[5], std_collector[5])