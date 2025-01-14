import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.focal_loss import FocalLoss, FocalLossAndCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerFocalLoss(nnUNetTrainer):
    """
    Standard nnUNetTrainer modified to have Focal Loss.
    """
    def _build_loss(self):
        loss = FocalLoss(gamma=2, smooth=1e-5, ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFLandCELoss(nnUNetTrainer):
    """
    Standard nnUNetTrainer modified to have a combination of Focal Loss
        and CrossEntropyLoss.
    """
    def _build_loss(self):
        loss = FocalLossAndCrossEntropyLoss(
            gamma=2,
            smooth=1e-5,
            ignore_index=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFLCELoss_StaticLR(nnUNetTrainerFLandCELoss):
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = None
        return optimizer, lr_scheduler
        
    def on_train_epoch_start(self):
        self.network.train()
        # self.lr_scheduler.step(self.current_epoch)  # Turn off lr scheduler
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)


class nnUNetTrainerFLCELossInitLR(nnUNetTrainerFLandCELoss):
    def __init__(
        self, 
        plans: dict, 
        configuration: str, 
        fold: int, 
        dataset_json: dict, 
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 5e-3  # Default is 1e-2

