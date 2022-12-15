from typing import Union, Tuple, List

import numpy as np
import torch
from torch.nn import functional as F


def resample_torch(data: Union[torch.Tensor, np.ndarray],
                   new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                   current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                   new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                   is_seg: bool = False):
    """
    Always runs on CPU

    equivalent to resample_skimage_simple with order 1 and antialiasing=false
    """
    new_shape = tuple(new_shape)
    with torch.no_grad():
        assert len(data.shape) == 4, "data must be c x y z"

        input_was_numpy = isinstance(data, np.ndarray)
        if input_was_numpy:
            data = torch.from_numpy(data)
        else:
            data = data.cpu()

        if is_seg:
            unique_values = torch.unique(data)
            result = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16)
            for i, u in enumerate(unique_values):
                result[i] = F.interpolate((data[None] == u).float(), new_shape, mode='trilinear', antialias=False)[0]
            result = unique_values[result.argmax(0)]
        else:
            result = F.interpolate(data[None].float(), new_shape, mode='trilinear', antialias=False)[0]
        if input_was_numpy:
            result = result.numpy()
    return result


if __name__ == '__main__':
    target_shape = (128, 128, 128)
    temp_data = torch.rand((2, 63, 63, 63))
    temp_seg = torch.randint(0, 4, (1, 63, 63, 63))
    ret_data = resample_torch(temp_data, target_shape, [], [], is_seg=False)
    ret_seg = resample_torch(temp_seg, target_shape, [], [], is_seg=True)
