import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
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
        for i in range(len(d[tag])):
            assert len(values[i][0]) == len(values[i][1])
            value_collection.append(values[i][1])

        df = pd.DataFrame(values[i][1], index=values[i][0], columns=[tag])
        df_collection.append(df)
    df_base = reduce(lambda x, y: pd.concat([x, y], axis=1, join="outer"), df_collection)
    # print(df_base)
    df_base.to_csv(os.path.join(path, 'summary.csv'))
    return df_base


def draw(df):
    target_v = [-0.08, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08]
    target_h = [0.10, 0.15, 0.20, 0.25]
    h_num = len(target_h)
    w_num = len(target_v)
    iterations = df.index
    fig, axes = plt.subplots(h_num, w_num, figsize=(h_num*5, w_num*5))
    axes = axes.flatten()

    def _draw(var, color, axes):
        cnt = 0
        for h in target_h:
            for v in target_v:
                for name in df.columns:
                    if v > 0 and str(-v) in name:
                        continue
                    if h > 0 and str(-h) in name:
                        continue
                    if str(v) in name and str(h) in name and var in name:
                        axes[cnt].plot(iterations, df[name], color, label=var)
                        axes[cnt].set_title(name.split('/')[0].split('loss_')[1])
                        # axes[cnt].set_ylim([0.0, 5.0])
                        cnt += 1
                        break
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=16)
    vars = ['task', 'velocity', 'height']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for var, color in zip(vars, colors):
        _draw(var, color, axes)
    fig.suptitle('Validation Loss', fontsize=20)
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = "../saved_results/sim_config_RL/DiffTaichi_RL/0702_011543/validation"
    df = to_dataframe(path)
    draw(df)

