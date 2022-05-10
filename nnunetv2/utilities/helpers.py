import torch


def softmax_helper(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)
