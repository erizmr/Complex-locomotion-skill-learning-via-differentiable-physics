import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import math


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        # if key not in self._data.keys():
        #     new_data = pd.DataFrame(index=[key], columns=['total', 'counts', 'average'])
        #     self._data = self._data.append(new_data)
        # self._data.total[key] += value * n
        # self._data.counts[key] += n
        # self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# def inf_loop(data_loader):
#     ''' wrapper function for endless data loader. '''
#     for loader in repeat(data_loader):
#         yield from loader

#
# def prepare_device(n_gpu_use):
#     """
#     setup GPU device if available. get gpu device indices which are used for DataParallel
#     """
#     n_gpu = torch.cuda.device_count()
#     if n_gpu_use > 0 and n_gpu == 0:
#         print("Warning: There\'s no GPU available on this machine,"
#               "training will be performed on CPU.")
#         n_gpu_use = 0
#     if n_gpu_use > n_gpu:
#         print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
#               "available on this machine.")
#         n_gpu_use = n_gpu
#     device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
#     list_ids = list(range(n_gpu_use))
#     return device, list_ids
#
#
#
#
# def unravel_indices(
#         indices: torch.LongTensor,
#         shape: Tuple[int, ...],
# ) -> torch.LongTensor:
#     r"""Converts flat indices into unraveled coordinates in a target shape.
#
#     Args:
#         indices: A tensor of (flat) indices, (*, N).
#         shape: The targeted shape, (D,).
#
#     Returns:
#         The unraveled coordinates, (*, N, D).
#     """
#
#     coord = []
#
#     for dim in reversed(shape):
#         coord.append(indices % dim)
#         indices = indices // dim
#
#     coord = torch.stack(coord[::-1], dim=-1)
#
#     return coord
#
#
# def get_gaussian_kernel(kernel_size=3, sigma=2, dim=2):
#     """
#     We assume it's a cube/square
#     :param kernel_size:
#     :param sigma:
#     :param dim:
#     :return:
#     """
#     # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
#     x_coord = torch.arange(kernel_size)
#     x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
#     y_grid = x_grid.t()
#     xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
#     assert (dim == 2 or dim == 3)
#     grid = torch.zeros([kernel_size for _ in range(dim)] + [dim])
#     if dim == 2:
#         for i in range(kernel_size):
#             for j in range(kernel_size):
#                 grid[i, j] = torch.tensor([i, j])
#     elif dim == 3:
#         for i in range(kernel_size):
#             for j in range(kernel_size):
#                 for k in range(kernel_size):
#                     grid[i, j, k] = torch.tensor([i, j, k])
#
#     mean = (kernel_size - 1) / 2.
#     variance = sigma ** 2.
#
#     # Calculate the 2-dimensional gaussian kernel which is
#     # the product of two gaussian distributions for two different
#     # variables (in this case called x and y)
#     gaussian_kernel = (1. / (2. * math.pi * variance)) * \
#                       torch.exp(
#                           -torch.sum((grid - mean) ** 2., dim=-1) / \
#                           (2 * variance)
#                       )
#
#     # Make sure sum of values in gaussian kernel equals 1.
#     gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
#
#     gaussian_kernel = gaussian_kernel.unsqueeze(dim)
#
#     gaussian_kernel = gaussian_kernel.repeat(*([1 for _ in range(dim)] + [dim]))
#
#     return gaussian_kernel
#
