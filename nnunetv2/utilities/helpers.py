import os
import sys

import torch


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def is_optimized_module(module) -> bool:
    """isinstance(module, torch._dynamo.OptimizedModule) without importing torch._dynamo (~0.4 s per process): a
    compiled module cannot exist unless torch._dynamo was already imported."""
    dynamo = sys.modules.get('torch._dynamo')
    return dynamo is not None and isinstance(module, dynamo.OptimizedModule)


def set_default_inductor_cache_dir():
    """
    Call right before torch.compile. torch.compile caches its compiled artifacts on disk, by default in
    /tmp/torchinductor_<user>, which a reboot wipes and which is node-local on clusters, so every process starts
    cold (several seconds for a 3d nnU-Net). With a persistent per-user default later processes start warm.

    An explicitly set TORCHINDUCTOR_CACHE_DIR always wins. Importing torch._dynamo writes inductor's own /tmp default
    into that variable, so a value equal to that default counts as unset. Inductor reads the variable at every cache
    access, so setting it late (after the import, before the first compiled call) is enough.
    """
    current = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
    if current is not None:
        try:
            from torch._inductor.runtime.cache_dir_utils import default_cache_dir
        except ImportError:
            return
        if os.path.abspath(current) != os.path.abspath(default_cache_dir()):
            return
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(base, 'nnunetv2', 'torchinductor')
