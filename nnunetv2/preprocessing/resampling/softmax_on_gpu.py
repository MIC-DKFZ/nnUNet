import os
from typing import Union, Tuple, List

import numpy as np
import torch

from torch.nn import functional as F


def maybe_resample_on_gpu(predicted_softmax: Union[np.ndarray, torch.Tensor],
                          new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                          *args, device: str = 'cuda') -> \
        Union[np.ndarray, torch.Tensor]:
    """
    tries to linearly resample predicted_softmax on the CPU. Failing that (OOM) it will fall back to CPU

    args are just there for compatibility with resampling function api
    """

    if not isinstance(predicted_softmax, torch.Tensor):
        predicted_softmax = torch.from_numpy(predicted_softmax)
        input_was_numpy = True
        input_device = None
    else:
        input_was_numpy = False
        input_device = predicted_softmax.device

    try:
        with torch.no_grad():
            softmax_resampled = torch.zeros((predicted_softmax.shape[0], *new_shape), dtype=torch.float,
                                            device=device)
            if not predicted_softmax.device == torch.device(device):
                softmax_gpu = predicted_softmax.to(torch.device(device))
            else:
                softmax_gpu = predicted_softmax
            for c in range(len(predicted_softmax)):
                softmax_resampled[c] = \
                    F.interpolate(softmax_gpu[c][None, None], size=new_shape, mode='trilinear')[0, 0]
            torch.cuda.empty_cache()

        if input_was_numpy:
            return softmax_resampled.cpu().numpy()
        else:
            return softmax_resampled.to(input_device)

    except RuntimeError as e:
        # clear cache of failed resampling attempts
        torch.cuda.empty_cache()
        os.environ['OMP_NUM_THREADS'] = '16'
        n_threads = torch.get_num_threads()
        torch.set_num_threads(16)
        torch.cuda.empty_cache()

        # gpu failed, try CPU
        print(f"\nGPU RESAMPLING FAILED: {e}\n")

        if not predicted_softmax.device == torch.device('cpu'):
            softmax_cpu = predicted_softmax.to(torch.device('cpu')).float()
        else:
            softmax_cpu = predicted_softmax

        with torch.no_grad():
            softmax_resampled = torch.zeros((predicted_softmax.shape[0], *new_shape), dtype=torch.float)
            for c in range(len(predicted_softmax)):
                softmax_resampled[c] = \
                    F.interpolate(softmax_cpu[c][None, None], size=new_shape, mode='trilinear')[0, 0]

        torch.set_num_threads(n_threads)

        if input_was_numpy:
            return softmax_resampled.numpy()
        else:
            return softmax_resampled.to(input_device)
