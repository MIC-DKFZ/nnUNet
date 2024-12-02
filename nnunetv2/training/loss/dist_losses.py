from typing import Callable

import torch
import numpy as np
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from torch import nn


class MemoryEfficientDistDiceLoss(MemoryEfficientSoftDiceLoss):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.0,
                 ddp: bool = True):
        super(MemoryEfficientDistDiceLoss, self).__init__(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth, ddp=ddp)

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, dist_map, loss_mask=None):
        dist_weight = torch.exp(-dist_map)  # Solves vanishing gradient

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = (y_onehot * dist_weight).sum(axes) if loss_mask is None else (y_onehot * loss_mask * dist_weight).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot * dist_weight).sum(axes)
            sum_pred = (x * dist_weight).sum(axes)
        else:
            intersect = (x * loss_mask * y_onehot).sum(axes)
            sum_pred = (x * loss_mask * dist_weight).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, min=1e-8))

        dc = dc.mean()
        return -dc