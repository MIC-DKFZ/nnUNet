import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerDiceLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = SoftDiceLoss(**{'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                    'do_bg': self.label_manager.has_regions, 'smooth': 1e-5, 'ddp': self.is_ddp},
                            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceLossClip1(nnUNetTrainer):
    def _build_loss(self):
        loss = SoftDiceLoss(**{'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                    'do_bg': self.label_manager.has_regions, 'smooth': 1e-5, 'ddp': self.is_ddp},
                            clip_tp=1, apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceLossLS01(nnUNetTrainer):
    def _build_loss(self):
        loss = SoftDiceLoss(**{'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                    'do_bg': self.label_manager.has_regions, 'smooth': 1e-5, 'ddp': self.is_ddp,
                               'label_smoothing': 0.1},
                            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceCELossLS01(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions aint working here for now'
        loss = DC_and_CE_loss({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp, 'label_smoothing': 0.1},
                              {'label_smoothing': 0.1},
                              weight_ce=1, weight_dice=1,
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


class nnUNetTrainerDiceCELossClip1(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp, 'clip_tp': 1},
                                   use_ignore_label=self.label_manager.ignore_label is not None)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp, 'clip_tp': 1},
                                  {}, weight_ce=1, weight_dice=1,
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


