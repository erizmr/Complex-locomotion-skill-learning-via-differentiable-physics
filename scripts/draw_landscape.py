import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt


# ret_paths = "/home/mingrui/difftaichi/difftaichi2/saved_results/sim_config_DiffPhy_robot5_vhc_5_l01_landscape/DiffTaichi_DiffPhy/0915_163915/validation/summary.csv"
# ret_paths = "/home/ljcc/repo/difftaichi2/saved_results/sim_config_DiffPhy_robot4_vh_landscape/DiffTaichi_DiffPhy/0804_030617/validation/summary.csv"
# ret_paths = "/home/ljcc/repo/difftaichi2/saved_results/sim_config_DiffPhy_robot5_vh_landscape/DiffTaichi_DiffPhy/0804_053055/validation/summary.csv"
ret_paths = "/home/ljcc/repo/difftaichi2/saved_results/sim_config_DiffPhy_robot3_vh_landscape/DiffTaichi_DiffPhy/0716_191831/validation/summary.csv"
ret_path = glob.glob(ret_paths)[-1]
print(ret_path)
ret = pd.read_csv(ret_path)
ret['Unnamed: 0'] = [name.split('_')[0] for name in ret['Unnamed: 0']]
ret.set_index('Unnamed: 0', inplace=True)
print(ret)
ret = ret.groupby(['Unnamed: 0']).mean()
print(ret)
losses = ret.loc["task"].values
print(losses)
one_way_sample_points = 50
heatmap = losses.copy().astype("float").reshape([one_way_sample_points * 2 + 1, one_way_sample_points * 2 + 1])
print(heatmap.shape)
print(heatmap[one_way_sample_points, one_way_sample_points])
sns.heatmap(heatmap)
# max_loss, min_loss = heatmap.max(), heatmap.min()
# print(max_loss, min_loss)
# levels = [(i / 10) * (max_loss - min_loss) + max_loss for i in range(11)]
# x = y = np.arange(-50, 51, 1)
# X, Y = np.meshgrid(x, y)
# plt.contourf(X, Y, heatmap, origin = "lower")
# plt.contour(X, Y, heatmap, colors = list("rgbcmy"), origin = "lower")
plt.show()

