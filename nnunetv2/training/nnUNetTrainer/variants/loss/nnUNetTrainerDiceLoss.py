import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerDiceLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = MemoryEfficientSoftDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
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

