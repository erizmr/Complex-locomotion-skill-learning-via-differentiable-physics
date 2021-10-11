import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.ndimage.filters import gaussian_filter1d
from collections import OrderedDict

robot_names = {2:"Alpaca", 3:"Monster", 4:"HugeStool", 5:"Stool", 6:"Snake"}
h_num = 1
w_num = 4
fig, axes = plt.subplots(h_num, w_num, figsize=(w_num * 7, h_num * 6))
axes = axes.flatten()

all_tanh_folders = glob.glob("./saved_results/sim_config_DiffPhy_robot*_vhc_5_l10_tanh/DiffTaichi_DiffPhy")
print(all_tanh_folders)

path_dict = {}
for path in all_tanh_folders:
    print(path)
    bs = int(path.split('robot')[-1].split('_')[0])
    print(bs)
    path_dict[bs] = glob.glob(os.path.join(path, "*"))
    print(path_dict)
data_dict = {}
for k, ps in path_dict.items():
    data_dict[k] = []
    for p in ps:
        ret = pd.read_csv(os.path.join(p, "validation/summary.csv"))
        ret['Unnamed: 0'] = [name.split('_')[0] for name in ret['Unnamed: 0']]
        ret.set_index('Unnamed: 0', inplace=True)
        # print(ret)
        ret = ret.groupby(['Unnamed: 0']).mean()
        data_dict[k].append(ret)

all_ours_folders = glob.glob("./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01/DiffTaichi_DiffPhy")
ours_path_dict = {}
for path in all_ours_folders:
    print(path)
    bs = int(path.split('robot')[-1].split('_')[0])
    print(bs)
    path_dict[bs] = glob.glob(os.path.join(path, "*"))
    print(path_dict)
ours_data_dict = {}
for k, ps in path_dict.items():
    ours_data_dict[k] = []
    for p in ps:
        ret = pd.read_csv(os.path.join(p, "validation/summary.csv"))
        ret['Unnamed: 0'] = [name.split('_')[0] for name in ret['Unnamed: 0']]
        ret.set_index('Unnamed: 0', inplace=True)
        # print(ret)
        ret = ret.groupby(['Unnamed: 0']).mean()
        ours_data_dict[k].append(ret)

normalize = True
save = True

list1 = list((x, y) for x, y in data_dict.items())
list2 = list((x, y) for x, y in ours_data_dict.items())
list1.sort()
list2.sort()


for (k, vals), (k2,vals2) in zip(data_dict.items(), ours_data_dict.items()):
    X = [int(x) for x in vals[0].columns]

    combined = np.array([v.loc["task"] for v in vals])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    base_loss = max(mean)
    ysmoothed = gaussian_filter1d(mean / base_loss, sigma=0.8)
    axes[k-2].plot(X, ysmoothed, label=f"Tanh")
    axes[k-2].fill_between(X, ysmoothed - std, ysmoothed + std, alpha=0.2)

    combined = np.array([v.loc["task"] for v in vals2])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    base_loss = max(mean)
    ysmoothed = gaussian_filter1d(mean / base_loss, sigma=0.8)
    axes[k-2].plot(X, ysmoothed, label=f"Sin")
    axes[k-2].fill_between(X, ysmoothed - std, ysmoothed + std, alpha=0.2)


    axes[k-2].set_xlabel("Iteartions", fontsize=20)
    axes[k-2].set_ylabel("Normlized Validation Loss", fontsize=20)
    axes[k-2].set_title(f"{robot_names[int(k)]}", fontsize=20)
    axes[k-2].tick_params(axis='x', labelsize=16)
    axes[k-2].tick_params(axis='y', labelsize=16)
    axes[k-2].legend(fontsize=18)


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
img_save_name = f"imgs/activation"
if save:
    with PdfPages(img_save_name + ".pdf") as pdf:
        pdf.savefig(fig)

plt.show()