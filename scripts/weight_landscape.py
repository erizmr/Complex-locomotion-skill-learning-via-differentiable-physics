import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import math

shape_list = []

def load_weights(name="save.pkl"):
    weight_list = []
    # print(name)
    w_val = pkl.load(open(name, 'rb'))
    for val in w_val:
        # print(val, val.shape)
        weight_list.append(val)
    return  weight_list


def flatten_weight(weight_list):
    return np.concatenate([x.flatten() for x in weight_list], axis=0)


def reshape_weight(weight_vec, shape_list):
    ret = []
    for shape in shape_list:
        item_num = math.prod(shape)
        item = weight_vec[:item_num].reshape(shape)
        ret.append(item)
    return ret


robot_id = 2
# folder_path = f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vha/DiffTaichi_DiffPhy/*/"
folder_path = f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vhc_5_l01/DiffTaichi_DiffPhy/*/"
# folder_path = f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vhc_5_l01_sgd/DiffTaichi_DiffPhy/*/"
# folder_path = f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vh/DiffTaichi_DiffPhy/*/"
# folder_path = f"./saved_results/sim_config_DiffPhy_robot{robot_id}_vhc_5_l01_batch_1/DiffTaichi_DiffPhy/*/"


do_normlize = False
exp_folder = glob.glob(folder_path)[-1]
print(f"experiment folder: {exp_folder}")
all_models = glob.glob(os.path.join(exp_folder, "models/iter*.pkl"))
a = [(int(x.split('/')[-1][4:-4]), x) for x in all_models]
model_paths = [y for x, y in sorted(a)]
print("models ", model_paths)
print("Path nums:", len(model_paths))
stacked_model_name = folder_path.split('/')[2]

flatten_weight_list = []
for m in model_paths:
    w = load_weights(m)
    if len(shape_list) == 0:
        for v in w:
            shape_list.append(v.shape)
    flatten_weight_list.append(flatten_weight(w))

print("shape list", shape_list)

stacked_weights = np.stack(flatten_weight_list, axis=0)

start_point = stacked_weights[0]
end_point = stacked_weights[-1]
optimal_from_start = end_point - start_point
print("Optimal from start", optimal_from_start.shape)

# weight_val_list = np.sum(stacked_weights, axis=1)
print(stacked_weights.shape)

for i in range(stacked_weights.shape[0]):
    # stacked_weights[i] = stacked_weights[i] - np.dot(np.dot(stacked_weights[i], optimal_from_start), optimal_from_start / np.linalg.norm(optimal_from_start))
    stacked_weights[i] = stacked_weights[i] - np.dot(np.dot(stacked_weights[i], optimal_from_start / np.linalg.norm(optimal_from_start)), optimal_from_start / np.linalg.norm(optimal_from_start))
print("norm", np.linalg.norm(optimal_from_start))
means = np.mean(stacked_weights, axis=0)
print("mean ", means.shape)

# Do normlize
stacked_weights -= means
pca_pipeline = make_pipeline(PCA(n_components=2))

pca_pipeline.fit(stacked_weights)
pca = pca_pipeline.named_steps['pca']

X = pca.transform(stacked_weights)
weights_optimal_normlized = stacked_weights[-1]
print("weight optimal shape", weights_optimal_normlized.shape)

eigenvalues = pca.explained_variance_
print(f"eigenvalues: {eigenvalues}, explained variance ratio: {pca.explained_variance_ratio_}")
# vec_x, vec_y = pca.components_  # Get eigenvector
vec_y, _ = pca.components_  # Get eigenvector
print(pca.components_.shape)

one_way_sample_points = 50
os.makedirs("validate_heatmap_weights", exist_ok=True)
weights = []

vec_x = optimal_from_start

vec_x = vec_x / np.linalg.norm(vec_x)
vec_y = vec_y / np.linalg.norm(vec_y)
print("Eigen vectors norm", np.linalg.norm(vec_x), np.linalg.norm(vec_y))

step_x = np.linalg.norm(optimal_from_start)
step_y = max(np.abs(X[:, 0]))

print(f"step x : {step_x}, step y: {step_y}")

for i in range(-one_way_sample_points, one_way_sample_points + 1):
    for j in range(-one_way_sample_points, one_way_sample_points + 1):
        new_weights_vec = weights_optimal_normlized + (i * vec_x * step_x + j * vec_y * step_y) / one_way_sample_points + means
        weight = reshape_weight(new_weights_vec, shape_list)
        weights.append(weight)
pkl.dump(weights, open(os.path.join("validate_heatmap_weights", "{}.pkl".format(stacked_model_name)), "wb"))
# print(X.shape)
x = X[:, 0]
y = X[:, 1]
steps = [x for x in range(X.shape[0])]
plt.plot(x, y, '.')
for i, txt in enumerate(steps):
    plt.annotate(txt, (x[i], y[i]))
plt.show()
