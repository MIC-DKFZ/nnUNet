from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.focal_loss import FocalLoss_Ori, Focal_and_DC_Loss
from nnunetv2.training.loss.focal_loss_2 import FocalLoss, MULTICLASS_MODE, MULTILABEL_MODE
from nnunetv2.training.loss.focal_loss_3 import FocalLossJunMa11
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np

from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerFocalLoss2(nnUNetTrainer):
    def _build_loss(self):
        loss = FocalLoss(MULTILABEL_MODE if self.label_manager.has_regions else MULTICLASS_MODE,
                         ignore_index=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFocalLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions
        loss = FocalLoss_Ori(self.label_manager.num_segmentation_heads, ignore_index=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFocalandDiceLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "region-based training not supported here"
        # loss = FocalLoss_Ori(num_class=self.label_manager.num_segmentation_heads, ignore_index=self.label_manager.ignore_label)
        loss = Focal_and_DC_Loss({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                 {'num_class': self.label_manager.num_segmentation_heads,
                                  'ignore_index': self.label_manager.ignore_label},
                                 ignore_label=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFocalLoss3(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "region-based training not supported here"
        assert not self.label_manager.has_ignore_label, "ignore label not supported here"

        loss = FocalLossJunMa11(apply_nonlin=softmax_helper_dim1)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

