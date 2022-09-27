import numpy as np

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.rmi_loss import RMILoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerRMILoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions
        assert not self.label_manager.has_ignore_label
        assert len(self.plans['configurations'][self.configuration]['patch_size']) == 2, "This only works with 2D data!"
        loss = RMILoss(self.label_manager.num_segmentation_heads)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
