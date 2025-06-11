import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class UnifiedFocalLoss(nn.Module):
    """Unified Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        # Convert to probabilities
        pred_prob = torch.softmax(pred, dim=1)

        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()

        # Focal weight
        p_t = torch.where(target_one_hot == 1, pred_prob, 1 - pred_prob)
        focal_weight = (1 - p_t) ** self.gamma

        # Dice loss component
        intersection = torch.sum(pred_prob * target_one_hot, dim=(2, 3, 4))
        union = torch.sum(pred_prob, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        # Cross entropy component
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        # Combine with focal weighting
        focal_ce = focal_weight.mean(dim=1) * ce_loss

        return self.alpha * focal_ce.mean() + (1 - self.alpha) * dice_loss


class TverskyLoss(nn.Module):
    """Tversky Loss for precision-recall balance"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super().__init__()
        self.alpha = alpha  # Weight for false negatives (recall)
        self.beta = beta    # Weight for false positives (precision)
        self.smooth = smooth

    def forward(self, pred, target):
        pred_prob = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()

        tp = torch.sum(pred_prob * target_one_hot, dim=(2, 3, 4))
        fn = torch.sum((1 - pred_prob) * target_one_hot, dim=(2, 3, 4))
        fp = torch.sum(pred_prob * (1 - target_one_hot), dim=(2, 3, 4))

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining segmentation and classification"""
    def __init__(self, seg_weight=1.0, cls_weight=0.5, loss_type='dice_ce', ddp=False):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

        # Segmentation losses
        if loss_type == 'dice_ce':
            self.dice_loss = MemoryEfficientSoftDiceLoss(
                apply_nonlin=softmax_helper_dim1,
                batch_dice=True,
                do_bg=False,
                smooth=1e-5,
                ddp=ddp,
            )
            self.ce_loss = RobustCrossEntropyLoss()
        elif loss_type == 'focal':
            self.seg_loss = UnifiedFocalLoss()
        elif loss_type == 'tversky':
            self.seg_loss = TverskyLoss()

        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.loss_type = loss_type

    def forward(self, outputs, targets):
        """
        outputs: dict with 'segmentation' and 'classification' keys
        targets: dict with 'segmentation' and 'classification' keys
        """
        seg_pred = outputs['segmentation']
        cls_pred = outputs['classification']
        seg_target = targets['segmentation']
        cls_target = targets['classification']

        # Segmentation loss
        if self.loss_type == 'dice_ce':
            seg_dice = self.dice_loss(seg_pred, seg_target)
            seg_ce = self.ce_loss(seg_pred, seg_target)
            seg_loss = seg_dice + seg_ce
        else:
            seg_loss = self.seg_loss(seg_pred, seg_target)

        # Classification loss
        cls_loss = self.cls_loss(cls_pred, cls_target)

        # Combined loss
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

        return {
            'total_loss': total_loss,
            'segmentation_loss': seg_loss,
            'classification_loss': cls_loss
        }