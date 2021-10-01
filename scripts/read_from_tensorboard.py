import os
from collections import defaultdict
import glob
import pandas as pd
from functools import reduce
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt

import numpy as np

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

for robot in range(5, 6):
    paths = f"./saved_results/sim_config_DiffPhy_robot{robot}_vh/DiffTaichi_DiffPhy/*/log"
    path = glob.glob(paths)
    path.sort(key=os.path.getmtime)
    for path_ in path:
        df = to_dataframe(path_)
        arr = df["TNS/train"].values
        # arr = np.log(arr)
        arr = list(arr)

        # arr.sort()

        plt.plot(list(range(len(arr))), arr)
        plt.show()