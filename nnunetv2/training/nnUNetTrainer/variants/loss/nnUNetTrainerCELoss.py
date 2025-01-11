import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
import numpy as np


class nnUNetTrainerCELoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = RobustCrossEntropyLoss(
            weight=None, ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerCELoss_5epochs(nnUNetTrainerCELoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5
        
class nnUNetTrainerCELoss_300epochs(nnUNetTrainerCELoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 300
