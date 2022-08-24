import numpy as np
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import DC_and_BCE_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_DiceUseClip(nnUNetTrainer):
    def _build_loss(self):
        raise DeprecationWarning('DC_and_CE_loss2 (using clip instead of +1e-8) is now standard')
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                    'do_bg': True, 'smooth': 1e-5}, use_ignore_label=self.label_manager.ignore_label is not None)
        else:
            loss = DC_and_CE_loss2({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                   'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1, weight_dice=1,
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


class nnUNetTrainer_DiceUseClip_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        raise DeprecationWarning('DC_and_CE_loss2 (using clip instead of +1e-8) is now standard. Also this trainer '
                                 'performed worse, so we need smooth')

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                    'do_bg': True, 'smooth': 0}, use_ignore_label=self.label_manager.ignore_label is not None)
        else:
            loss = DC_and_CE_loss2({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                   'smooth': 0, 'do_bg': False}, {}, weight_ce=1, weight_dice=1,
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