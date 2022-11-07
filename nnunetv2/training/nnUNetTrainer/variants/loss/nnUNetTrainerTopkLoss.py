from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
from nnunetv2.training.loss.robust_ce_loss import TopKLoss


class nnUNetTrainerTopk10Loss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = TopKLoss(ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100,
                        k=10)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerTopk10LossLS01(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = TopKLoss(ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100,
                        k=10, label_smoothing=0.1)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceTopK10Loss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = DC_and_topk_loss({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                {'k': 10,
                                 'label_smoothing': 0.0},
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
