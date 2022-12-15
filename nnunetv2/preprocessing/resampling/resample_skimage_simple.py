from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from nnunetv2.preprocessing.resampling.resample_torch import resample_torch


def resample_skimage_simple(data: Union[torch.Tensor, np.ndarray],
                   new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                   current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                   new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                   is_seg: bool = False,
                   anti_aliasing: bool = False,  # only applies if is_seg=False
                   order: int = 1
                   ):
    new_shape = tuple(new_shape)
    assert len(data.shape) == 4, "data must be c x y z"
    if is_seg:
        unique_values = np.sort(pd.unique(data.ravel()))
        result = np.zeros((data.shape[0], *new_shape), dtype=data.dtype)
        # we can only resample 3d data, do not resample 4d data here
        for c in range(data.shape[0]):
            res_here = np.zeros((len(unique_values), *new_shape), dtype=np.float16)
            for i, u in enumerate(unique_values):
                res_here[i] = resize((data[c] == u).astype(float), new_shape, order=order, mode='edge',
                                     anti_aliasing=False)
            result[c] = unique_values[res_here.argmax(0)]
    else:
        result = np.zeros((data.shape[0], *new_shape), dtype=data.dtype)
        for c in range(data.shape[0]):
            result[c] = resize(data[c].astype(np.float32), new_shape, order=order, mode='edge', anti_aliasing=anti_aliasing)
    return result


if __name__ == '__main__':
    from time import time
    target_shape = (256, 256, 256)
    temp_data = gaussian_filter(np.random.random((2, 127, 127, 127)), sigma=3) * 300
    temp_seg = np.random.randint(0, 4, (1, 127, 127, 127))
    st = time()
    for i in range(1):
        ret_data = resample_torch(temp_data, target_shape, [], [], is_seg=False)
    print(f'data torch {time() - st}')
    st = time()
    for i in range(1):
        ret_seg = resample_torch(temp_seg, target_shape, [], [], is_seg=True)
    print(f'seg torch {time() - st}')
    st = time()
    for i in range(1):
        ret_data_sk = resample_skimage_simple(temp_data, target_shape, [], [], is_seg=False, order=1, anti_aliasing=False)
    print(f'data skimage {time() - st}')
    st = time()
    for i in range(1):
        ret_seg_sk = resample_skimage_simple(temp_seg, target_shape, [], [], is_seg=True, order=1, anti_aliasing=False)
    print(f'seg skimage {time() - st}')

    print(np.all(np.isclose(ret_data, ret_data_sk)))

    from time import time
    target_shape = (512, 512, 512)
    temp_seg = np.random.randint(0, 7, (1, 31, 127, 61))
    st = time()
    for i in range(1):
        ret_seg = resample_torch(temp_seg, target_shape, [], [], is_seg=True)
    print(f'seg torch {time() - st}')
    st = time()
    for i in range(1):
        ret_seg_sk = resample_skimage_simple(temp_seg, target_shape, [], [], is_seg=True, order=1, anti_aliasing=False)
    print(f'seg skimage {time() - st}')

    print(np.all(np.isclose(ret_seg, ret_seg_sk)))
