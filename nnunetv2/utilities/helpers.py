import torch

try:
    from torch import GradScaler  # torch >= 2.3
    TORCH_HAS_OLD_GRADSCALER = False
except ImportError:
    from torch.cuda.amp import GradScaler  # torch < 2.3
    TORCH_HAS_OLD_GRADSCALER = True

AMP_DEVICE_TYPES = ('cuda', 'xpu')


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'xpu':
        torch.xpu.empty_cache()
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


def uses_amp(device: torch.device) -> bool:
    return device.type in AMP_DEVICE_TYPES


def make_grad_scaler(device: torch.device):
    if not uses_amp(device):
        return None
    if TORCH_HAS_OLD_GRADSCALER:
        return GradScaler()
    return GradScaler(device.type)


def autocast_if_available(device: torch.device):
    if uses_amp(device):
        return torch.autocast(device.type, enabled=True)
    return dummy_context()
