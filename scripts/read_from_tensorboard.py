import os
from collections import defaultdict
import glob
import pandas as pd
from functools import reduce
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

import numpy as np

robot_names = {2:"Alpaca", 3:"Monster", 4:"HugeStool", 5:"Stool"}
save = True

def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath) if ".log" not in dname]
    tags = defaultdict(list)
    for i, it in enumerate(summary_iterators):
        for tg in it.Tags()['scalars']:
            tags[tg].append(i)

    out = defaultdict(list)

    for tag, index in tags.items():
        if 'steps_per_sec' in tag:
            continue
        for ind in index:
            value = []
            step = []
            for event in summary_iterators[ind].Scalars(tag):
                step.append(event.step)
                value.append(event.value)
            out[tag].append([step, value])
    return out


def to_dataframe(dpath):
    d = tabulate_events(dpath)
    value_collection = []
    df_collection = []
    for tag, values in d.items():
        # print("tag {} values {}".format(tag, values))
        selected_val = []
        selected_index = []
        for i in range(len(d[tag])):
            assert len(values[i][0]) == len(values[i][1])
            # value_collection.append(values[i][1])
            if len(values[i][0]) > len(selected_val):
                selected_val = values[i][1]
                selected_index = values[i][0]
            # print(tag, selected_index, len(values[i][0]))

        df = pd.DataFrame(selected_val, index=selected_index, columns=[tag])
        df_collection.append(df)
    print("current data path", dpath, len(df_collection))
    df_base = reduce(lambda x, y: pd.concat([x, y], axis=1, join="outer"), df_collection)
    # print(df_base)
    # df_base.to_csv(os.path.join(dpath, 'summary_old.csv'))
    return df_base

max_values = {}
all_valuse = defaultdict(list)
for robot in range(2, 6):
    # paths = f"./saved_results/sim_config_DiffPhy_robot{robot}_vh/DiffTaichi_DiffPhy/*/log"
    paths = f"./saved_results/sim_config_DiffPhy_robot{robot}_vhc_5_l01/DiffTaichi_DiffPhy/*/log"
    path = glob.glob(paths)
    path.sort(key=os.path.getmtime)
    max_values[robot] = 0.0
    for path_ in path:
        df = to_dataframe(path_)
        arr = df["TNS/train"].values
        # print(arr.shape)
        # arr = np.log(arr)
        arr = list(arr)
        all_valuse[robot].extend(arr)
        arr_max = max(arr)

        if arr_max > max_values[robot]:
            max_values[robot] = arr_max
        # arr.sort()
        # plt.plot(list(range(len(arr))), arr)
        # plt.show()
print(max_values)
# print(all_valuse)
print(len(all_valuse[2]))

fig, axes = plt.subplots(1, 4, figsize=(4 * 7, 6))
axes = axes.flatten()

for i in range(2, 6):
    print(len(all_valuse[i]))
    axes[i-2].hist(np.log(all_valuse[i]), bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85, density=True)
    axes[i-2].set_title(f"{robot_names[i]}", fontsize=28)
    axes[i-2].set_ylabel(f"Probability Density ", fontsize=24)
    axes[i - 2].set_xlabel(f"Sum of Gradients Norm (log scale)", fontsize=24)
    axes[i-2].tick_params(axis='x', labelsize=20)
    axes[i-2].tick_params(axis='y', labelsize=20)
    axes[i-2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axes[i-2].legend(fontsize=14)
plt.tight_layout()
fig = plt.gcf()
img_save_name = f"imgs/gradient_analysis"
if save:
    with PdfPages(img_save_name + ".pdf") as pdf:
        pdf.savefig(fig)
plt.show()