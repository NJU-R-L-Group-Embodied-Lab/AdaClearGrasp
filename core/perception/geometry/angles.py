# perception/geometry/angles.py
import numpy as np


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def to_numpy(data):
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.array(data)


def safe_index(tensor_or_array, index):
    if hasattr(tensor_or_array, "dim"):  # torch
        return tensor_or_array[index] if tensor_or_array.dim() > 1 else tensor_or_array
    return tensor_or_array[index] if tensor_or_array.ndim > 1 else tensor_or_array
