from typing import Callable

import torch
import numpy as np
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.focal_loss import FocalLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from time import time

class SoftDistDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None, dist_weight_func= lambda x: torch.exp(-x)):

        super(SoftDistDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp
        self.dist_weight_func = dist_weight_func

    def forward(self, x, y, dist_map, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        dist_weight = dist_map if self.dist_weight_func is not None else self.dist_weight_func(dist_map)

        tp, fp, fn, _ = get_voxel_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        dist_tp = tp * dist_weight
        dist_fp = fp * dist_weight
        dist_fn = fn * dist_weight

        dist_tp = dist_tp.sum(dim=axes, keepdim=False)
        dist_fp = dist_fp.sum(dim=axes, keepdim=False)
        dist_fn = dist_fn.sum(dim=axes, keepdim=False)

        if self.ddp and self.batch_dice:
            dist_tp = AllGatherGrad.apply(dist_tp).sum(0)
            dist_fp = AllGatherGrad.apply(dist_fp).sum(0)
            dist_fn = AllGatherGrad.apply(dist_fn).sum(0)

        if self.clip_tp is not None:
            dist_tp = torch.clip(dist_tp, min=self.clip_tp , max=None)

        nominator = 2 * dist_tp
        denominator = 2 * dist_tp + dist_fp + dist_fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc
    
class DistTverskyLoss(SoftDistDiceLoss):
    def __init__(self, apply_nonlin: Callable = None, batch_tversky: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None, dist_weight_func = lambda x: x+1):

        super(DistTverskyLoss, self).__init__(apply_nonlin, batch_tversky, do_bg, smooth, ddp, clip_tp, dist_weight_func)

    def forward(self, x, y, dist_map, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        dist_weight = dist_map if self.dist_weight_func is not None else self.dist_weight_func(dist_map)

        tp, fp, fn, _ = get_voxel_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        dist_fp = fp * dist_weight
        dist_fn = fn * dist_weight

        tp = tp.sum(dim=axes, keepdim=False)
        dist_fp = dist_fp.sum(dim=axes, keepdim=False)
        dist_fn = dist_fn.sum(dim=axes, keepdim=False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            dist_fp = AllGatherGrad.apply(dist_fp).sum(0)
            dist_fn = AllGatherGrad.apply(dist_fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + dist_fp + dist_fn

        tversky = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky

class MemoryEfficientDistDiceLoss(MemoryEfficientSoftDiceLoss):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.0,
                 ddp: bool = True, dist_weight_func=lambda x: torch.exp(-x)):
        super(MemoryEfficientDistDiceLoss, self).__init__(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth, ddp=ddp)

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.dist_weight_func = dist_weight_func

    def forward(self, x, y, dist_map, loss_mask=None):
        dist_weight = dist_map if self.dist_weight_func is not None else self.dist_weight_func(dist_map)

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
            intersect = (x * loss_mask * y_onehot * dist_weight).sum(axes)
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
class DistDSC_and_CE_loss(nn.Module):
    def __init__(self, dist_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientDistDiceLoss):
        super(DistDSC_and_CE_loss, self).__init__()

        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dist_dc = dice_class(apply_nonlin=softmax_helper_dim1, **dist_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, dist_map: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dist_dc_loss = self.dist_dc(net_output, target_dice, dist_map, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dist_dc_loss
        return result

class DistDSC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, dist_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientDistDiceLoss):
        super(DistDSC_and_BCE_loss, self).__init__()

        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dist_dc = dice_class(apply_nonlin=softmax_helper_dim1, **dist_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, dist_map: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dist_dc_loss = self.dist_dc(net_output, target, dist_map, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dist_dc_loss
        return result

class SoftDistDSC_and_CE_loss(DistDSC_and_CE_loss):
    def __init__(self, dist_tversky_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        super(SoftDistDSC_and_CE_loss, self).__init__(dist_tversky_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label,
                                                      dice_class=SoftDistDiceLoss)

class SoftDistDSC_and_BCE_loss(DistDSC_and_BCE_loss):
    def __init__(self, bce_kwargs, dist_tversky_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False):
        super(SoftDistDSC_and_BCE_loss, self).__init__(bce_kwargs, dist_tversky_kwargs, weight_ce, weight_dice, use_ignore_label,
                                                       dice_class=SoftDistDiceLoss)

class DistTversky_and_CE_loss(DistDSC_and_CE_loss):
    def __init__(self, dist_tversky_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        super(DistTversky_and_CE_loss, self).__init__(dist_tversky_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label,
                                                      dice_class=DistTverskyLoss)

class DistTversky_and_BCE_loss(DistDSC_and_BCE_loss):
    def __init__(self, bce_kwargs, dist_tversky_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False):
        super(DistTversky_and_BCE_loss, self).__init__(bce_kwargs, dist_tversky_kwargs, weight_ce, weight_dice, use_ignore_label,
                                                       dice_class=DistTverskyLoss)

class DistDSC_and_Focal_loss(nn.Module):
    def __init__(self, dist_dice_kwargs, focal_kwargs, weight_focal=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientDistDiceLoss):
        super(DistDSC_and_Focal_loss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        self.focal = FocalLoss(**focal_kwargs)
        self.dist_dc = dice_class(apply_nonlin=softmax_helper_dim1, **dist_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, dist_map: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DistDSC_and_Focal_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dist_dc_loss = self.dist_dc(net_output, target_dice, dist_map, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        focal_loss = self.focal(net_output, target[:, 0]) \
            if self.weight_focal != 0 and (self.ignore_label is None or num_fg > 0) else 0


        result = self.weight_focal * focal_loss + self.weight_dice * dist_dc_loss
        return result

def get_voxel_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return: tp, fp, fn, tn without summing
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    return tp, fp, fn, tn