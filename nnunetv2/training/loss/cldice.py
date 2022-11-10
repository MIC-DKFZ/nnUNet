import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert3DTo2DTransform, \
    Convert2DTo3DTransform
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_, slice_wise: bool = False):
    if slice_wise:
        shp = img.shape
        assert len(shp) == 5, 'only supports 5D tensors (b, c, x, y, z)'
        tr_fwd = Convert3DTo2DTransform(('img', )) # orig_shape_{k}
        reshaped = tr_fwd(img=img)
        img = reshaped['img']
        del reshaped

    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)

    if slice_wise:
        r_bwd = Convert2DTo3DTransform(('skel', ))
        reshaped = r_bwd(skel=skel, orig_shape_skel=shp)
        skel = reshaped['skel']
    return skel


def soft_cldice(y_true, y_pred, iters: int = 3, smooth_term: float = 1, loss_mask: torch.Tensor = None,
                slicewise: bool = False):
    assert len(y_pred.shape) == len(y_true.shape)
    skel_pred = soft_skel(y_pred, iters, slice_wise=slicewise)
    skel_true = soft_skel(y_true, iters, slice_wise=slicewise)
    if loss_mask is not None:
        skel_true *= loss_mask
        skel_pred *= loss_mask
    tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + smooth_term) / (
                torch.sum(skel_pred[:, 1:, ...]) + smooth_term)
    tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + smooth_term) / (
                torch.sum(skel_true[:, 1:, ...]) + smooth_term)
    cl_dice = - 2.0 * (tprec * tsens) / (tprec + tsens)
    return cl_dice


class Dc_BCE_clDice_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_clDice: float = 1,
                 iters: int = 3, smooth_clDice: float = 1., ignore_label=None, cldice_slicewise: bool = False):
        super().__init__()
        "no support for ignore label. Also no region based training"
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label
        self.cldice_slicewise = cldice_slicewise
        self.iters = iters
        self.smooth_clDice = smooth_clDice

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_clDice = weight_clDice

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=lambda x: x, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = (target != self.ignore_label).float()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        y_onehot = torch.zeros(net_output.shape, device=net_output.device)
        y_onehot.scatter_(1, target_dice.long(), 1)
        del target_dice

        pred_softmaxed = softmax_helper_dim1(net_output)
        cldice_loss = soft_cldice(y_onehot, pred_softmaxed, iters=self.iters, smooth_term=self.smooth_clDice,
                                  loss_mask=mask, slicewise=self.cldice_slicewise) \
            if self.weight_clDice != 0 else 0
        dc_loss = self.dc(pred_softmaxed, y_onehot, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        del pred_softmaxed

        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_clDice * cldice_loss
        return result
