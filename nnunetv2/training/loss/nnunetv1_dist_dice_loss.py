import torch
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, SoftDiceLossSquared, DC_and_CE_loss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np
import time as time
import edt

ANISOTROPY = (0.096, 0.096, 0.096)

def get_dist_tp_fp_fn_tn(net_output, gt, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared
    :return: Confusion matrix Tensors scaled by voxel-wise distance maps for each class

    NOTE: This is different from get_tp_fp_fn_tn, which returns the summed values.

    Heavily influenced by: https://github.com/seung-lab/euclidean-distance-transform-3d
    """
    shp_x = net_output.shape
    shp_y = gt.shape
    num_batches = shp_x[0]
    num_classes = shp_x[1]

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            # gt is likely (b, x, y(, z))
            gt = gt.view((num_batches, 1, *shp_x[2:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            # gt is likely (b, 1, x, y(, z))
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

        # gt_resized is of shape (b*x, y(, z)) since batches not supported in edt
        # gt_resized = gt.squeeze()
        # if num_batches > 1: gt_resized = torch.cat((*gt_resized[:,...],))
        # assert gt_resized.shape[0] == num_batches * shp_x[2], 'Batches were not concatenated properly!'
        
        dt = np.zeros(shp_y) # (b, 1, x, y(, z))
        gt_resized = gt.squeeze().cpu().numpy() + 1 # +1 to turn background into foreground segmentation for distance mapping
        if num_batches > 1:
            for b in range(num_batches):
                dt[b,0,...] = edt.edt(gt_resized[b,...], anisotropy=ANISOTROPY, parallel=0) + 1
        else: dt[0,0,...] = edt.edt(gt_resized, anisotropy=ANISOTROPY, parallel=0) + 1

    dt = torch.nan_to_num(torch.from_numpy(dt), nan=1.0, posinf=1.0, neginf=1.0)
    if net_output.device.type == "cuda":
        dt = dt.cuda(net_output.device.index)

    # dt is resized from (b*x, y(, z)) to (b, 1, x, y(, z))
    # dt = torch.stack(torch.split(dt, num_batches*[shp_x[2]], dim=0), dim=0).unsqueeze(dim=1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    fp = fp * dt
    fn = fn * dt

    return tp, fp, fn, tn

class DistancePenalization(nn.Module):
    def __init__(self, apply_nonlin=None, do_bg=False, smooth=1e-5):
        """
        """
        super(DistancePenalization, self).__init__()

        self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        _, fp, fn, _ = get_dist_tp_fp_fn_tn(x, y, loss_mask, False)
        
        if not self.do_bg:
            if self.batch_dice:
                fp = fp[1:]
                fn = fn[1:]
            else:
                fp = fp[:, 1:]
                fn = fn[:, 1:]
                
        fp = torch.mean(fp)
        fn = torch.mean(fn)

        return fp + fn + self.smooth

class DistDiceLoss(SoftDiceLoss):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-5):
        """
        Distance map penalized Dice loss for multiclass segmentation
        Motivated by: https://openreview.net/forum?id=B1eIcvS45V
        Distance Map Loss Penalty Term for Semantic Segmentation
        Adapted from the Binary version: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/boundary_loss.py
        """
        super(DistDiceLoss, self).__init__(apply_nonlin, batch_dice, do_bg, smooth)

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_dist_tp_fp_fn_tn(x, y, loss_mask, False)

        if len(axes) > 0:
            tp = sum_tensor(tp, axes, keepdim=False)
            fp = sum_tensor(fp, axes, keepdim=False)
            fn = sum_tensor(fn, axes, keepdim=False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

class DistDC_and_CE_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Distance-Mapped Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__(soft_dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                         log_dice, ignore_label)

        self.dc = DistDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        start = time.time()
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        print(f"Dist Dice: {-dc_loss}")
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        end = time.time()
        print(f"Dist Dice Time: {end-start} seconds")
        return result

class DC_CE_DP_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 weight_dp=1, log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE, Dice and distance penalization do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        :param weight_dp:
        """
        super().__init__(soft_dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                         log_dice, ignore_label)

        self.dp = DistancePenalization(apply_nonlin=softmax_helper)
        self.weight_dp = weight_dp

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        start = time.time()
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        dp_loss = self.dp(net_output, target) if self.weight_dp != 0 else 0

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_dp * dp_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        end = time.time()
        print(f"Loss Calculation Time: {end-start}")
        return result