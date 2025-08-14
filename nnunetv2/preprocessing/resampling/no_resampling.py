from typing import Union, Tuple, List

import numpy as np
import torch


def no_resampling_hack(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]
):
    return data