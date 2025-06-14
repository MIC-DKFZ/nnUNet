"""
Stage 1 multi-task trainer: Train encoder + segmentation, freeze classification head
"""

import torch
import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask


class nnUNetTrainerMultiTaskFrozenClassifier(nnUNetTrainerMultiTask):
    """
    Stage 1 trainer: Trains encoder + segmentation while freezing classification head
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.print_to_log_file("=== Stage 1: Encoder + Segmentation Training ===")
        self.print_to_log_file("Classification head will be frozen")
        self.print_to_log_file("Only encoder and segmentation will be trained")

        # Override loss weights for stage 1
        self.segmentation_weight = 1.0
        self.classification_weight = 0.0  # Turn off classification loss

        self.print_to_log_file(f"Loss weights - Seg: {self.segmentation_weight}, Cls: {self.classification_weight}")

    def initialize(self):
        """Override to freeze classification head after initialization"""
        super().initialize()
        self._freeze_classification_head()
        self._verify_parameter_setup()

    def _freeze_classification_head(self):
        """Freeze classification head parameters"""
        frozen_count = 0
        for name, param in self.network.named_parameters():
            if 'classification_head' in name:
                param.requires_grad = False
                frozen_count += 1
                self.print_to_log_file(f"Frozen (cls): {name}")

        self.print_to_log_file(f"Frozen {frozen_count} classification parameters")

    def _verify_parameter_setup(self):
        """Verify parameter freezing"""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        self.print_to_log_file("=== Stage 1 Parameter Setup ===")
        self.print_to_log_file(f"Total: {total_params:,}")
        self.print_to_log_file(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        self.print_to_log_file(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")

    def compute_loss(self, output, target, loss_dict=None):
        """Override to disable classification loss"""
        if loss_dict is None:
            loss_dict = {}

        # Only compute segmentation loss
        seg_output = output['segmentation']
        seg_target = target['segmentation']
        seg_loss = super().compute_loss(seg_output, seg_target)
        loss_dict['seg_loss'] = seg_loss * self.segmentation_weight

        # Set classification loss to zero
        loss_dict['cls_loss'] = torch.tensor(0.0, device=self.device)

        # Total loss is just segmentation loss
        total_loss = loss_dict['seg_loss']
        loss_dict['total_loss'] = total_loss

        return total_loss

    def configure_optimizers(self):
        """Configure optimizer for only trainable (encoder + segmentation) parameters"""
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]

        self.print_to_log_file(f"Stage 1 optimizer configured for {len(trainable_params)} parameter groups")

        # Use full learning rate for encoder + segmentation training
        optimizer = torch.optim.SGD(trainable_params, self.initial_lr,
                                   weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, self.num_epochs, power=0.9)

        self.print_to_log_file(f"Stage 1 learning rate: {self.initial_lr}")

        # Store for use in training loop
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        """Override to ensure proper optimizer/scheduler order"""
        # Call parent train_step which handles the actual training
        result = super().train_step(batch)

        # The lr_scheduler.step() should be called after optimizer.step()
        # This is typically handled in the main training loop, but we can ensure it here
        return result

    # Remove the on_train_epoch_end method if it exists to avoid scheduler conflicts