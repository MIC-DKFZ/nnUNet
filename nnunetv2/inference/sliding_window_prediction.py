import numpy as np
import torch
from typing import Union, Tuple, List

from torch import nn
from torch.nn import functional as F





def predict_3d_image_with_3d_sliding_window(network: nn.Module,
                                            input_image: Union[np.ndarray, torch.Tensor],
                                            patch_size: Tuple[int, int, int],
                                            mirror_axes: Tuple[int, ...] = None,
                                            step_size: float = 0.5,
                                            use_gaussian: bool = True,
                                            precomputed_gausian: torch.Tensor = None,
                                            perform_everything_on_gpu: bool = True,
                                            verbose: bool = True) -> Union[np.ndarray, torch.Tensor]:
    assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'
    if verbose: print("step_size:", step_size)
    if verbose: print("mirror_axes:", mirror_axes)
    data, slicer = pad_nd_image(input_image, patch_size, 'constant', {'constant_values': 0}, True, None)
    data_shape = data.shape  # still c, x, y, z


if __name__ == '__main__':
    a = torch.rand((4, 2, 32, 23))
    a_npy = a.numpy()

    a_padded = pad_nd_image(a, new_shape=(13, 27))
    a_npy_padded = pad_nd_image(a_npy, new_shape=(13, 27))
    assert all([i == j for i, j in zip(a_padded.shape, (4, 2, 48, 27))])
    assert all([i == j for i, j in zip(a_npy_padded.shape, (4, 2, 48, 27))])
    assert np.all(a_padded.numpy() == a_npy_padded)


