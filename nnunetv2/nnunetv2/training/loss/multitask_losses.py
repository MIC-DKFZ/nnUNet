import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import os

DEBUG=os.environ.get("DEBUG", False)

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
    def __init__(self, seg_weight=1.0, cls_weight=0.5, loss_type='dice_ce', ddp=False, ignore_label=None):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.ignore_label = ignore_label

        # Segmentation losses - EXACTLY like baseline
        if loss_type == 'dice_ce':
            soft_dice_kwargs = {
                'batch_dice': True,
                'do_bg': False,  # Don't include background in dice
                'smooth': 1e-5,
                'ddp': ddp,
            }
            ce_kwargs = {}
            if ignore_label is not None:
                ce_kwargs['ignore_index'] = ignore_label

            self.dice_loss = MemoryEfficientSoftDiceLoss(
                apply_nonlin=softmax_helper_dim1,
                **soft_dice_kwargs
            )
            self.ce_loss = RobustCrossEntropyLoss(**ce_kwargs)
        elif loss_type == 'focal':
            self.seg_loss = UnifiedFocalLoss()
        elif loss_type == 'tversky':
            self.seg_loss = TverskyLoss()

        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.loss_type = loss_type

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        if DEBUG:
            print(f"[LOSS DEBUG] Input shapes - Seg pred: {seg_pred.shape if not isinstance(seg_pred, list) else [x.shape for x in seg_pred]}")
            print(f"[LOSS DEBUG] Seg target: {seg_target.shape}, Cls pred: {cls_pred.shape}, Cls target: {cls_target.shape}")
            print(f"[LOSS DEBUG] Seg pred range: [{seg_pred[0].min().item():.6f}, {seg_pred[0].max().item():.6f}]" if isinstance(seg_pred, list) else f"[{seg_pred.min().item():.6f}, {seg_pred.max().item():.6f}]")
            print(f"[LOSS DEBUG] Cls pred range: [{cls_pred.min().item():.6f}, {cls_pred.max().item():.6f}]")
            print(f"[LOSS DEBUG] Cls target unique values: {cls_target.unique()}")

        # Ensure classification predictions are within a reasonable range
        cls_pred = torch.clamp(cls_pred, min=-10.0, max=10.0)

        # Segmentation loss - FOLLOW BASELINE PATTERN EXACTLY
        if self.loss_type == 'dice_ce':
            # Handle ignore label like baseline
            if self.ignore_label is not None:
                assert seg_target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
                mask = (seg_target != self.ignore_label).bool()
                target_dice = torch.clone(seg_target)
                target_dice[seg_target == self.ignore_label] = 0
                num_fg = mask.sum()
            else:
                target_dice = seg_target
                mask = None
                num_fg = None

            # Dice loss with proper mask handling
            seg_dice = self.dice_loss(seg_pred, target_dice, loss_mask=mask) if self.seg_weight != 0 else 0

            # CE loss with proper target format (remove channel dimension)
            if seg_target.shape[1] == 1:  # Ensure target has channel dimension
                ce_target = seg_target[:, 0]  # Remove channel dimension like baseline
            else:
                ce_target = seg_target

            seg_ce = self.ce_loss(seg_pred, ce_target) if self.seg_weight != 0 and (self.ignore_label is None or num_fg > 0) else 0

            seg_loss = seg_dice + seg_ce

            if DEBUG:
                print(f"[LOSS DEBUG] Seg dice: {seg_dice.item() if isinstance(seg_dice, torch.Tensor) else seg_dice:.6f}")
                print(f"[LOSS DEBUG] Seg CE: {seg_ce.item() if isinstance(seg_ce, torch.Tensor) else seg_ce:.6f}")
        else:
            seg_loss = self.seg_loss(seg_pred, seg_target)
            if DEBUG:
                print(f"[LOSS DEBUG] Seg loss: {seg_loss.item():.6f}")

        # Classification loss
        if DEBUG:
            print(f"[LOSS DEBUG] Pre-classification cls pred sample: {cls_pred[0]}")

        cls_loss = self.cls_loss(cls_pred, cls_target)
        if DEBUG:
            print(f"[LOSS DEBUG] Raw classification loss: {cls_loss.item():.6f}")

        # Combined loss - weighted like baseline
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

        if DEBUG:
            print(f"[LOSS DEBUG] Final losses - Seg: {seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss:.6f}, Cls: {cls_loss.item():.6f}, Total: {total_loss.item():.6f}")
            print(f"[LOSS DEBUG] Loss weights - Seg: {self.seg_weight}, Cls: {self.cls_weight}")

        return {
            'loss': total_loss,
            'segmentation_loss': seg_loss,
            'classification_loss': cls_loss
        }