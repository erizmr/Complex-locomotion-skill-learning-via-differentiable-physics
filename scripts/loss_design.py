import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.ndimage.filters import gaussian_filter1d

all_batching_folders = glob.glob("./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01_naive_loss/DiffTaichi_DiffPhy")
all_batching_folders = glob.glob("./saved_results/sim_config_DiffPhy_robot*_vhc_5_l01/DiffTaichi_DiffPhy")
print(all_batching_folders)
plt.figure(figsize=(7, 6))
path_dict = {}
# naive loss
for path in all_batching_folders:
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

normalize = True
save = True
# if normlize:
#     base_loss = df[name][0]
# else:
#     base_loss = 1.0

for k, vals in data_dict.items():
    X = [int(x) for x in vals[0].columns]
    combined = np.array([v.loc["velocity"] for v in vals])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    base_loss = max(mean)
    ysmoothed = gaussian_filter1d(mean / base_loss, sigma=0.9)
    plt.plot(X, ysmoothed, label=f"Agent {k}")
    plt.fill_between(X, ysmoothed - std, ysmoothed + std, alpha=0.2)
plt.xlabel("Iteartions", fontsize=20)
plt.ylabel("Normlized Validation Loss", fontsize=20)
plt.title("Loss Design Ablation", fontsize=20)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
# plt.tight_layout(rect=(0, 0.04, 1.0, 1.0))
plt.tight_layout()
# plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()

labels_dict = dict(sorted(zip(labels, handles), key=lambda t: int(t[0].split(" ")[-1])))
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0].split(" ")[-1])))
print(labels_dict.keys())
plt.legend( handles, labels, fontsize=14, loc="upper right")

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
img_save_name = f"imgs/loss_design"
if save:
    with PdfPages(img_save_name + ".pdf") as pdf:
        pdf.savefig(fig)

plt.show()



