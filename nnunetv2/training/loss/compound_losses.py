import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.fpr_loss import SoftFPRLoss, MemoryEfficientSoftFPRLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import logging

class DC_CE_FNR_loss(nn.Module):
    
    def __init__(self,
                 soft_dice_kwargs,
                 ce_kwargs,
                 soft_fpr_kwargs=None,
                 weight_ce=1,
                 weight_dice=1,
                 weight_fpr=1,
                 ignore_label=None,
                 dice_class=SoftDiceLoss,
                 fpr_class=SoftFPRLoss):

        super().__init__()

        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_fpr = weight_fpr
        self.ignore_label = ignore_label
        soft_fpr_kwargs = soft_fpr_kwargs or {}

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        # Only create FPR if weight > 0
        if weight_fpr > 0:
            self.fpr = fpr_class(apply_nonlin=softmax_helper_dim1, **soft_fpr_kwargs)
        else:
            self.fpr = None
    def forward(self, net_output: torch.Tensor, target: torch.Tensor, return_components=False):

        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_clean = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_clean = target
            mask = None
    
        # Dice
        dc_loss = self.dc(net_output, target_clean, loss_mask=mask) \
            if self.weight_dice != 0 else 0
    
        # CE
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
    
        # FPR
        if self.weight_fpr != 0:
            fpr_loss = self.fpr(net_output, target_clean, loss_mask=mask)
        else:
            fpr_loss = 0
    
        result = (
            self.weight_ce * ce_loss +
            self.weight_dice * dc_loss +
            self.weight_fpr * fpr_loss
        )
        if return_components:
            return result, {
                "ce": ce_loss.detach(),
                "dice": dc_loss.detach(),
                "fpr": fpr_loss.detach()
            }

    
        return result

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1,weight_fpr=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss,soft_fpr_kwargs=None,fpr_class=MemoryEfficientSoftFPRLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label
        soft_fpr_kwargs = soft_fpr_kwargs or {}

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        self.weight_fpr = weight_fpr

        if weight_fpr > 0:
            self.fpr = fpr_class(apply_nonlin=torch.sigmoid, **soft_fpr_kwargs)
        else:
            self.fpr = None


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
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

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        fpr_loss = self.fpr(net_output, target_regions, loss_mask=mask) \
        if self.weight_fpr != 0 else 0
    
        result = (
            self.weight_ce * ce_loss +
            self.weight_dice * dc_loss +
            self.weight_fpr * fpr_loss
        )
        logging.info(f"CE: {ce_loss.item():.4f}, Dice: {dc_loss.item():.4f}, FPR: {fpr_loss.item():.4f}")
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self,
                 soft_dice_kwargs,
                 ce_kwargs,
                 soft_fpr_kwargs=None,
                 weight_ce=1,
                 weight_dice=1,
                 weight_fpr=0,
                 ignore_label=None):

        super().__init__()

        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_fpr = weight_fpr
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        soft_fpr_kwargs = soft_fpr_kwargs or {}

        if weight_fpr > 0:
            self.fpr = SoftFPRLoss(apply_nonlin=softmax_helper_dim1, **soft_fpr_kwargs)
        else:
            self.fpr = None

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = (target != self.ignore_label).bool()
            target_clean = torch.clone(target)
            target_clean[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_clean = target
            mask = None
            num_fg = None

        # Dice
        dc_loss = self.dc(net_output, target_clean, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        # TopK CE
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # FPR
        fpr_loss = self.fpr(net_output, target_clean, loss_mask=mask) \
            if (self.fpr is not None and self.weight_fpr != 0) else 0

        total_loss = (
            self.weight_ce * ce_loss +
            self.weight_dice * dc_loss +
            self.weight_fpr * fpr_loss
        )
        logging.info(f"CE: {ce_loss.item():.4f}, Dice: {dc_loss.item():.4f}, FPR: {fpr_loss.item():.4f}")
        return total_loss
