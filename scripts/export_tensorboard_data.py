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
    df_base = reduce(lambda x, y: pd.concat([x, y], axis=1, join="outer"), df_collection)
    # print(df_base)
    df_base.to_csv(os.path.join(dpath, 'summary.csv'))
    return df_base


def draw(df_dict, robot_id):
    target_v = [-0.08, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08]
    target_h = [0.10, 0.15, 0.20]
    h_num = len(target_h)
    w_num = len(target_v)
    for k, v in df_dict.items():
        iterations = v.index
    # fig, axes = plt.subplots(h_num, w_num, figsize=(h_num*5, w_num*5),  constrained_layout=True)
    fig, axes = plt.subplots(h_num, w_num, figsize=(h_num * 5, w_num * 5))
    axes = axes.flatten()

    def _draw(df, tag, var, color, alpha, axes):
        cnt = 0
        for h in target_h:
            for v in target_v:
                for name in df.columns:
                    if v > 0 and str(-v) in name:
                        continue
                    if h > 0 and str(-h) in name:
                        continue
                    if str(v) in name and str(h) in name and var in name.split("_loss")[0]:
                        axes[cnt].plot(iterations, df[name], color, label=tag+" "+var+" loss", alpha=alpha)
                        axes[cnt].set_title(name.split('/')[0].split('loss_')[1])
                        # axes[cnt].set_ylim([0.0, 5.0])
                        cnt += 1
                        break
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=16)

    vars = ['task', 'velocity', 'height']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    alpha = 1.0
    for tag, df in df_dict.items():
        for var, color in zip(vars, colors):
            _draw(df, tag, var, color, alpha, axes)
        alpha *= 0.5

    fig.suptitle(f'Validation Loss - Robot {robot_id}', fontsize=20)
    # fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=10.0)
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # path_ours = "saved_results/sim_config_DiffPhy_with_actuation_robot5/DiffTaichi_DiffPhy/0714_012455/validation"
    import argparse
    parser = argparse.ArgumentParser(description='draw')
    parser.add_argument('--our_file_path',
                        default='',
                        help='experiment tensorboard file')
    parser.add_argument('--rl_file_path',
                        default='',
                        help='experiment tensorboard file')
    parser.add_argument('--no-rl',
                        action='store_true',
                        help='experiment tensorboard file')
    args = parser.parse_args()

    import glob
    path_ours = glob.glob(os.path.join(args.our_file_path, "*/validation"))
    print(path_ours)
    path_ours = sorted(path_ours, key=os.path.getmtime)[-1]
    print("Path ours", path_ours)
    robot_id = path_ours.split("_robot")[-1].split('/')[0]
    df_ours = to_dataframe(path_ours)

    if not args.no_rl:
        path_rls = glob.glob(os.path.join(args.rl_file_path, "*/validation"))
        print(path_rls)
        path_rl = sorted(path_rls, key=os.path.getmtime)[0]
        print("Path rls", path_rl)
        # path_ppo = "saved_results/sim_config_RL/DiffTaichi_RL/0702_011543/validation"
        df_ppo = to_dataframe(path_rl)

    if args.no_rl:
        df_dict = {"Ours": df_ours}
    else:
        df_dict = {"Ours": df_ours, "PPO": df_ppo}

    draw(df_dict, robot_id)

