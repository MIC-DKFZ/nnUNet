from tkinter import FALSE
import torch
from torch import nn, Tensor
import numpy as np
import os

DEBUG=os.environ.get("DEBUG", "False")

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if DEBUG:
            print(f"[CE DEBUG] Input shapes - input: {input.shape}, target: {target.shape}")
            print(f"[CE DEBUG] Input range: [{input.min().item():.6f}, {input.max().item():.6f}]")
            print(f"[CE DEBUG] Target unique values: {target.unique()}")
            print(f"[CE DEBUG] Target range: [{target.min().item()}, {target.max().item()}]")

        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]

        result = super().forward(input, target)

        if DEBUG:
            print(f"[CE DEBUG] Computed CE loss: {result.item():.6f}")

        return result


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
