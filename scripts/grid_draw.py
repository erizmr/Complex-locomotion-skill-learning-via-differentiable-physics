import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.ndimage.filters import gaussian_filter1d

robot_names = {2:"Alpaca", 3:"Monster", 4:"HugeStool", 5:"Stool", 6:"Snake"}
name_substitute = {"velocity":"running", "height":"jumping", "crawl":"crawling", "task":"task"}

robot_data_dfs = []
for i in range(2, 6):
    with open(f"df_{i}.pkl", "rb") as f:
        robot_data_dfs.append(pkl.load(f))


# for df in robot_data_dfs:
#     print(df)
normlize = True
error_bar = True
save = True
task = "tvh"
fig, axs = plt.subplots(1, len(robot_data_dfs), figsize=(5*len(robot_data_dfs), 5.5))
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

draw_len = 100000
cnt = 0
for cnt, df_dict in enumerate(robot_data_dfs):
    alpha = 1.0
    for k, df in df_dict.items():
        draw_len = min(len(df.index.values), draw_len)
    for k, df in df_dict.items():
        line_style = "-"
        if k == "PPO":
            line_style = "--"
        iterations = df.index.values
        for i, name in enumerate(vars):
            base_loss = 1.
            if normlize:
                base_loss = df[name].values[0]
            ysmoothed = gaussian_filter1d(df[name][:draw_len] / base_loss, sigma=1.0)
            axs[cnt].plot(iterations[:draw_len], ysmoothed, color=colors[i], label=k + " " + name_substitute[name] + " loss",
                    alpha=alpha, linestyle=line_style, dashes=(5, 3) if line_style == "--" else (None, None))
            if error_bar:
                axs[cnt].fill_between(iterations[:draw_len],
                                ysmoothed - df[name + '_std'][:draw_len],
                                ysmoothed + df[name + '_std'][:draw_len],
                                color=colors[i], alpha=0.2)
        alpha *= 0.8
    if normlize:
        axs[cnt].set_ylabel("Normlized Validation Loss", fontsize=20)
    else:
        axs[cnt].set_ylabel("Loss", fontsize=20)
    axs[cnt].set_xlabel("Training Iterations", fontsize=20)
    axs[cnt].set_title(f'"{robot_names[cnt+2]}"', fontsize=28)
    axs[cnt].tick_params(axis='x', labelsize=16)
    axs[cnt].tick_params(axis='y', labelsize=16)

fig.tight_layout(rect=(0, 0.08, 1.0, 1.0))
labels_handles = {
  label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
}

print(labels_handles.keys())
print(labels_handles)
fig.legend(
    labels_handles.keys(),
    # loc="lower left",
loc=(0.0, 0.0),
    ncol=6,
    mode="expand",
    handlelength=2.5,
    columnspacing=3.0,
    edgecolor='white',
    framealpha=0.,
    fontsize=20
)


img_save_name = f"imgs/validation_total_loss_all_robot"
if save:
    with PdfPages(img_save_name + ".pdf") as pdf:
        pdf.savefig(fig)

plt.show()


