from copy import deepcopy
from time import time
from typing import Union, Tuple, List

import SimpleITK
import numpy as np
import torch
from torch.nn import functional as F


def resample_torch(data: Union[torch.Tensor, np.ndarray],
                   new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                   current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                   new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                   is_seg: bool = False,
                   num_threads: int = 4,
                   device: torch.device = torch.device('cpu'),
                   memefficient_seg_resampling: bool = False):
    """
    equivalent to resample_skimage_simple with order 1 and antialiasing=false
    """
    if isinstance(new_shape, np.ndarray):
        new_shape = [int(i) for i in new_shape]

    if all([i == j for i, j in zip(new_shape, data.shape[1:])]):
        return data
    else:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads)
        new_shape = tuple(new_shape)
        with torch.no_grad():
            assert len(data.shape) == 4, "data must be c x y z"

            input_was_numpy = isinstance(data, np.ndarray)
            if input_was_numpy:
                data = torch.from_numpy(data).to(device)
            else:
                orig_device = deepcopy(data.device)
                data = data.to(device)

            if is_seg:
                unique_values = torch.unique(data)
                result_dtype = torch.int8 if max(unique_values) < 127 else torch.int16
                result_seg = torch.zeros((data.shape[0], *new_shape), dtype=result_dtype, device=device)
                if not memefficient_seg_resampling:
                    # believe it or not, the implementation below is 3x as fast (at least on Liver CT and on CPU)
                    # Why? Because argmax is slow. The implementation below immediately sets most locations and only lets the
                    # uncertain ones be determined by argmax

                    # unique_values = torch.unique(data)
                    # result = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16)
                    # for i, u in enumerate(unique_values):
                    #     result[i] = F.interpolate((data[None] == u).float() * 1000, new_shape, mode='trilinear', antialias=False)[0]
                    # result = unique_values[result.argmax(0)]

                    result = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16, device=device)
                    scale_factor = 1000
                    done_mask = torch.zeros_like(result_seg, dtype=torch.bool, device=device)
                    for i, u in enumerate(unique_values):
                        result[i] = F.interpolate((data[None] == u).float() * scale_factor, new_shape, mode='trilinear', antialias=False)[0]
                        mask = result[i] > (0.7 * scale_factor)
                        result_seg[mask] = u.item()
                        done_mask |= mask
                    if not torch.all(done_mask):
                        # print('resolving argmax', torch.sum(~done_mask), "voxels to go")
                        result_seg[~done_mask] = unique_values[result[:, ~done_mask].argmax(0)].to(result_dtype)
                    result = result_seg
                else:
                    for i, u in enumerate(unique_values):
                        if u == 0:
                            pass
                        result_seg[F.interpolate((data[None] == u).float(), new_shape, mode='trilinear', antialias=False)[0] > 0.5] = u
                    result = result_seg
            else:
                result = F.interpolate(data[None].float(), new_shape, mode='trilinear', antialias=False)[0]
            if input_was_numpy:
                result = result.cpu().numpy()
            else:
                result = result.to(orig_device)
        torch.set_num_threads(n_threads)
        return result


if __name__ == '__main__':
    torch.set_num_threads(16)
    target_shape = (128, 128, 128)
    temp_data = torch.rand((16, 63, 63, 63))
    temp_seg = torch.randint(0, 4, (16, 63, 63, 63))
    # temp_seg = torch.zeros((16, 63, 63, 63))
    temp_seg[:, :10, :10, :10] = 1
    temp_seg[:, :10, -10:, :10] = 2
    temp_seg[:, :10, :10, -10:] = 3
    print('cpu')
    st = time()
    # ret_data = resample_torch(temp_data, target_shape, [], [], is_seg=False, device=torch.device('cpu'))
    ret_seg = resample_torch(temp_seg, target_shape, [], [], is_seg=True, device=torch.device('cpu'))
    print(time() - st, 's')
    print('cpu memeff')
    st = time()
    ret_seg = resample_torch(temp_seg, target_shape, [], [], is_seg=True, device=torch.device('cpu'), memefficient_seg_resampling=True)
    print(time() - st, 's')
    print('gpu')
    st = time()
    ret_data = resample_torch(temp_data, target_shape, [], [], is_seg=False, device=torch.device('cuda:0'))
    ret_seg = resample_torch(temp_seg, target_shape, [], [], is_seg=True, device=torch.device('cuda:0'))
    print(time() - st, 's')

