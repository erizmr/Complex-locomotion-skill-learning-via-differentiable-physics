import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.backends.backend_pdf import PdfPages


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
    print("current data path", dpath, len(df_collection))
    df_base = reduce(lambda x, y: pd.concat([x, y], axis=1, join="outer"), df_collection)
    # print(df_base)
    df_base.to_csv(os.path.join(dpath, 'summary.csv'))
    return df_base


def draw(df_dict, robot_id, task, save=True, normlize=False, error_bar=False):

    target_v = [-0.08, -0.06, -0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08]
    target_h = [0.10, 0.15, 0.20]
    target_c = [0.0, 1.0]
    h_num = len(target_h)
    w_num = len(target_v)

    # vars = ['task', 'velocity', 'height', 'crawl']
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink']

    vars = []
    colors = []
    if 't' in task:
        vars.append('task')
        colors.append('tab:blue')
    if 'v' in task:
        vars.append('velocity')
        colors.append('tab:orange')
    if 'h' in task:
        vars.append('height')
        colors.append('tab:green')
    if 'c' in task:
        vars.append('crawl')
        colors.append('tab:pink')
        h_num += 1

    iterations = None
    for k, v in df_dict.items():
        iterations = v.index
    # fig, axes = plt.subplots(h_num, w_num, figsize=(h_num*5, w_num*5),  constrained_layout=True)
    fig, axes = plt.subplots(h_num, w_num, figsize=(w_num * 5, h_num * 5))
    axes = axes.flatten()

    def _draw(df, tag, var, color, alpha, axes, normlize=False, error_bar=False):
        cnt = 0
        if var == 'crawl':
            cnt = len(target_v) * len(target_h)
        for c in target_c:
            if var == 'crawl' and c < 1e-6:
                continue
            for h in target_h:
                if h > 0.1 and c > 0.0:
                    continue
                for v in target_v:
                    for name in df.columns:
                        if v > 0 and str(-v) in name:
                            continue
                        if h > 0 and str(-h) in name:
                            continue
                        if str(v) in name and str(h) in name and str(c) in name and var in name.split("_loss")[0]:
                            # print(var + "_" + str(c))
                            if var == 'crawl' and var + "_" + str(c) not in name:
                                continue
                            # print(f"Iter len {len(iterations)}, Data len {len(df[name])}")
                            draw_len = min(len(iterations), len(df[name]))
                            if normlize:
                                base_loss = df[name][0]
                            else:
                                base_loss = 1.0
                            axes[cnt].plot(iterations[:draw_len],
                                           df[name][:draw_len] / base_loss, color,
                                           label=tag+" "+var+" loss", alpha=alpha)
                            if error_bar:
                                axes[cnt].fill_between(iterations[:draw_len],
                                                 df[name][:draw_len] / base_loss - df[name+'_std'][:draw_len],
                                                 df[name][:draw_len] / base_loss + df[name+'_std'][:draw_len],
                                                 color=color, alpha=0.2)
                            # axes[cnt].errorbar(iterations[:draw_len], df[name][:draw_len] / base_loss,
                            #                    yerr=df[name+'_std'][:draw_len], color=color, ecolor=color,
                            #                    label=tag+" "+var+" loss", alpha=0.1)
                            axes[cnt].set_title(name.split('/')[0].split('loss_')[1], fontsize=12)
                            axes[cnt].set_aspect('auto')
                            # if normlize:
                            #     axes[cnt].set_ylim([0.0, 1.2])
                            cnt += 1
                            break
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=16)

    alpha = 1.0
    for tag, df in df_dict.items():
        for var, color in zip(vars, colors):
            _draw(df, tag, var, color, alpha, axes, normlize=normlize, error_bar=error_bar)
        alpha *= 0.5

    fig.suptitle(f'Validation Loss - Robot {robot_id}', fontsize=20)
    # fig.tight_layout(pad=0.0, w_pad=2.0, h_pad=5.0)
    # plt.tight_layout()
    img_save_name = f"imgs/validation_loss_robot_{robot_id}_{list(df_dict.keys())[-1]}"
    if save:
        with PdfPages(img_save_name + ".pdf") as pdf:
            pdf.savefig(fig)
    # plt.savefig(img_save_name, dpi=fig.dpi)
    plt.show()


def draw_single(df_dict, robot_id, task, save=True, normlize=False, error_bar=False):
    vars = []
    colors = []
    if 't' in task:
        vars.append('task')
        colors.append('tab:blue')
    if 'v' in task:
        vars.append('velocity')
        colors.append('tab:orange')
    if 'h' in task:
        vars.append('height')
        colors.append('tab:green')
    if 'c' in task:
        vars.append('crawl')
        colors.append('tab:pink')
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    alpha = 1.0
    draw_len = 100000
    for k, df in df_dict.items():
        draw_len = min(len(df.index.values), draw_len)
    for k, df in df_dict.items():
        iterations = df.index.values
        for i, name in enumerate(vars):
            base_loss = 1.
            if normlize:
                base_loss = df[name].values[0]
            ax.plot(iterations[:draw_len], df[name][:draw_len] / base_loss, color=colors[i], label=k+" "+name+" loss", alpha=alpha)
            if error_bar:
                ax.fill_between(iterations[:draw_len],
                                       df[name][:draw_len] / base_loss - df[name + '_std'][:draw_len],
                                       df[name][:draw_len] / base_loss + df[name + '_std'][:draw_len],
                                       color=colors[i], alpha=0.2)
        alpha *=0.5
    if normlize:
        ax.set_ylabel("Normlized Validation Loss")
    else:
        ax.set_ylabel("Loss")
    ax.set_xlabel("Iterations")
    ax.set_title(f"Validation total loss robot {robot_id}")
    plt.legend()
    img_save_name = f"imgs/validation_total_loss_robot_{robot_id}"
    if save:
        with PdfPages(img_save_name + ".pdf") as pdf:
            pdf.savefig(fig)
    plt.show()


def merge_df_single_plot(df_list, task):
    # df_new = merge_df_with_error_bar(df_list, std=False)
    df_new_list = []

    for df_new in df_list:
        iterations = df_new.index
        data_dict = defaultdict(list)
        for name in df_new.columns:
            tag = name.split('_')[0]
            if '0.25' in name or tag[0] not in task:
                continue
            data_dict[tag].append(df_new[name].values)
        key_list = list(data_dict.keys())
        for k in key_list:
            # print('k', k, np.array(data_dict[k]))
            data_dict[k] = np.mean(np.array(data_dict[k]), axis=0)
            # data_dict[k+'_std'] = np.std(np.array(data_dict[k]), axis=0)
        df_sub_ret = pd.DataFrame(data_dict)
        df_sub_ret.set_index(iterations.values, inplace=True)
        df_new_list.append(df_sub_ret)
    # print(df_new_list)
    df_ret = merge_df_with_error_bar(df_new_list)
    # print(df_ret)
    return df_ret


def merge_df_with_error_bar(df_list, std=True):
    df_new = pd.DataFrame()
    names = df_list[0].columns
    iterations = df_list[0].index
    # print(iterations)
    for n in names:
        if '0.25' in n:
            continue
        vals = []
        for df in df_list:
            # print(df[n].values.shape)
            vals.append(df[n].values)
        # print(np.array(vals).shape)
        df_new[n] = np.mean(np.array(vals), axis=0)
        if std:
            df_new[n+'_std'] = np.std(np.array(vals), axis=0)
        # print(df_new[n])
    df_new.set_index(iterations.values, inplace=True)
    print(df_new)
    return df_new


if __name__ == '__main__':
    # path_ours = "saved_results/sim_config_DiffPhy_with_actuation_robot5/DiffTaichi_DiffPhy/0714_012455/validation"
    import argparse
    parser = argparse.ArgumentParser(description='draw')
    parser.add_argument('--our_file_path',
                        default='',
                        help='DiffPhy experiment tensorboard file')
    parser.add_argument('--rl_file_path',
                        default='',
                        help='RL experiment tensorboard file')
    parser.add_argument('--task',
                        default='tvh',
                        help='task type, t: task total loss, v:velocity, h:height, c:crawl')
    parser.add_argument('--error-bar',
                        action='store_true',
                        help='draw error bar')
    parser.add_argument('--no-rl',
                        action='store_true',
                        help='exclude rl experiments')
    parser.add_argument('--no-diffphy',
                        action='store_true',
                        help='exclude rl experiments')
    parser.add_argument('--no-normlize',
                        action='store_true',
                        help='do not normlize the loss')
    parser.add_argument('--no-save',
                        action='store_true',
                        help='do not save the plots in pdf')
    parser.add_argument('--draw-single',
                        action='store_true',
                        help='do not save the plots in pdf')
    args = parser.parse_args()

    import glob
    df_ours = None
    robot_id_our = None
    if not args.no_diffphy:
        path_ours = glob.glob(os.path.join(args.our_file_path, "*/validation"))
        print(path_ours)
        path_ours = sorted(path_ours, key=os.path.getmtime)
        print("Path ours", path_ours)
        robot_id_our = path_ours[0].split("_robot")[-1].split('/')[0]
        # df_ours = to_dataframe(path_ours)
        df_ours_all = []
        for p in path_ours:
            df_ours_all.append(to_dataframe(p))
        df_ours = merge_df_with_error_bar(df_ours_all)
        df_ours_single = merge_df_single_plot(df_ours_all, args.task)
    # print(df_ours)

    df_ppo = None
    robot_id_rl = None
    if not args.no_rl:
        path_rls = glob.glob(os.path.join(args.rl_file_path, "*/validation"))
        print(path_rls)
        path_rls = sorted(path_rls, key=os.path.getmtime)
        print("Path rls", path_rls)
        robot_id_rl = path_rls[0].split("_robot")[-1].split('/')[0]
        # df_ppo = to_dataframe(path_rl)
        df_rl_all = []
        for p in path_rls:
            df_rl_all.append(to_dataframe(p))
        df_ppo = merge_df_with_error_bar(df_rl_all)
        df_ppo_single = merge_df_single_plot(df_rl_all, args.task)
    # else:
    #     robot_id_rl = robot_id_our
    # if not args.no_rl and not args.no_diffphy:
    #     assert robot_id_our == robot_id_rl
    #     print(robot_id_our, robot_id_rl)
    robot_id = robot_id_our if robot_id_our is not None else robot_id_rl
    print(robot_id)
    df_dict = None
    if args.no_rl:
        df_dict = {"Ours": df_ours}
    if args.no_diffphy:
        df_dict = {"PPO": df_ppo}
    if not args.no_rl and not args.no_diffphy:
        assert df_ppo is not None
        df_dict = {"Ours": df_ours, "PPO": df_ppo}

    if args.draw_single:
        df_dict_single = {"control length 5": df_ours_single, "control length 1": df_ppo_single}
        draw_single(df_dict_single,
                    robot_id,
                    args.task,
                    save=not args.no_save,
                    normlize=not args.no_normlize,
                    error_bar=args.error_bar)
    else:
        draw(df_dict, robot_id, args.task,
             save=not args.no_save,
             normlize=not args.no_normlize,
             error_bar=args.error_bar)
