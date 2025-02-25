import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.focal_loss import FocalLoss
from nnunetv2.training.loss.compound_losses import DC_and_Focal_loss
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
import numpy as np

class nnUNetTrainerFocalLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = FocalLoss(**{'alpha': 1, 'gamma': 2, 'reduction': 'mean'})
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

class nnUNetTrainerFocalLoss_300epochs(nnUNetTrainerFocalLoss):
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


class nnUNetTrainerDiceFocalLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = DC_and_Focal_loss({'batch_dice': self.configuration_manager.batch_dice,
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                 {'alpha': 1, 'gamma': 2, 'reduction': 'mean'},
                                  weight_focal=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
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

class nnUNetTrainerDiceFocalLoss_300epochs(nnUNetTrainerDiceFocalLoss):
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