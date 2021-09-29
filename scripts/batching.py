import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_batching_folders = glob.glob("./saved_results/sim_config_DiffPhy_robot2_vhc_5_l10_batch_*/DiffTaichi_DiffPhy")
print(all_batching_folders)

path_dict = {}
for path in all_batching_folders:
    print(path)
    bs = int(path.split('/')[-2].split('_')[-1])
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

# print(data_dict)
# if normlize:
#     base_loss = df[name][0]
# else:
#     base_loss = 1.0

for k, vals in data_dict.items():
    X = [int(x) for x in vals[0].columns]
    combined = np.array([v.loc["task"] for v in vals])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    plt.plot(X, mean, label=f"Batch size {k}")
    plt.fill_between(X, mean-std, mean+std, alpha=0.2)
plt.xlabel("Iteartion")
plt.ylabel("Validation Loss")
plt.title("Batch Size Ablation")
plt.legend()
plt.show()
