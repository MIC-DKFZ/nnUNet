import numpy as np

from nnunetv2.training.loss.cldice import Dc_BCE_clDice_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_clDice(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError()
        else:
            loss = Dc_BCE_clDice_loss({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                       'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=0.5,
                                      weight_dice=0.5, weight_clDice=1, iters=4, smooth_clDice=1e-5,
                                      ignore_label=self.label_manager.ignore_label, cldice_slicewise=False
                                      )

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainer_clDice_sliceWise(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError()
        else:
            loss = Dc_BCE_clDice_loss({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                       'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=0.5,
                                      weight_dice=0.5, weight_clDice=1, iters=4, smooth_clDice=1e-5,
                                      ignore_label=self.label_manager.ignore_label, cldice_slicewise=True
                                      )

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

