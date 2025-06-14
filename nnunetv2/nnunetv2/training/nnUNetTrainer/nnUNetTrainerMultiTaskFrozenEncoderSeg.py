"""
Stage 2 multi-task trainer: Load stage 1 weights, freeze encoder + segmentation, train classification
"""

import torch
import os
from os.path import join, isfile
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask


class nnUNetTrainerMultiTaskFrozenEncoderSeg(nnUNetTrainerMultiTask):
    """
    Stage 2 trainer: Loads stage 1 weights and trains only classification head
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.print_to_log_file("=== Stage 2: Classification Head Training ===")
        self.print_to_log_file("Encoder and segmentation will be frozen")
        self.print_to_log_file("Only classification head will be trained")

        # Override loss weights for stage 2
        self.segmentation_weight = 0.0  # Turn off segmentation loss
        self.classification_weight = 1.0

        self.print_to_log_file(f"Loss weights - Seg: {self.segmentation_weight}, Cls: {self.classification_weight}")

    def initialize(self):
        """Override to load stage 1 weights and freeze encoder + segmentation"""
        super().initialize()
        self._load_stage1_weights()
        self._freeze_encoder_and_segmentation()
        self._verify_parameter_setup()

    def _load_stage1_weights(self):
        """Load weights from stage 1 training (encoder + segmentation)"""
        stage1_path = self._find_stage1_checkpoint()

        self.print_to_log_file(f"Loading stage 1 weights from: {stage1_path}")

        checkpoint = torch.load(stage1_path, map_location=self.device, weights_only=False)
        pretrained_state_dict = checkpoint['network_weights']

        # Get current model state dict
        if hasattr(self.network, 'module'):
            current_state_dict = self.network.module.state_dict()
        else:
            current_state_dict = self.network.state_dict()

        # Load all parameters except classification head (keep it random for stage 2)
        loaded_count = 0
        skipped_count = 0

        for key, value in pretrained_state_dict.items():
            if 'classification_head' in key:
                self.print_to_log_file(f"Skipping cls head: {key}")
                skipped_count += 1
                continue

            if key in current_state_dict:
                if current_state_dict[key].shape == value.shape:
                    current_state_dict[key] = value
                    loaded_count += 1
                    self.print_to_log_file(f"Loaded: {key} {value.shape}")
                else:
                    self.print_to_log_file(f"Shape mismatch: {key}")
                    skipped_count += 1
            else:
                self.print_to_log_file(f"Key not found: {key}")
                skipped_count += 1

        # Apply updated state dict
        if hasattr(self.network, 'module'):
            self.network.module.load_state_dict(current_state_dict)
        else:
            self.network.load_state_dict(current_state_dict)

        self.print_to_log_file(f"Stage 1 -> Stage 2: Loaded {loaded_count}, skipped {skipped_count}")

    def _find_stage1_checkpoint(self):
        """Find the checkpoint from stage 1 training"""
        # Look for stage 1 results with the frozen classifier suffix
        current_output_folder = self.output_folder

        # Replace stage 2 suffix with stage 1 suffix
        stage1_output_folder = current_output_folder.replace('_frozen_encoder_seg', '_frozen_classifier')

        if stage1_output_folder == current_output_folder:
            # If no replacement happened, construct stage 1 path manually
            base_folder = current_output_folder.replace('nnUNetTrainerMultiTaskFrozenEncoderSeg', 'nnUNetTrainerMultiTaskFrozenClassifier')
            stage1_output_folder = base_folder

        best_checkpoint = join(stage1_output_folder, "checkpoint_best.pth")
        final_checkpoint = join(stage1_output_folder, "checkpoint_final.pth")

        self.print_to_log_file(f"Looking for stage 1 checkpoint in: {stage1_output_folder}")

        if isfile(best_checkpoint):
            return best_checkpoint
        elif isfile(final_checkpoint):
            return final_checkpoint
        else:
            raise RuntimeError(f"Could not find stage 1 checkpoint in {stage1_output_folder}. "
                             f"Please run stage 1 training first with nnUNetTrainerMultiTaskFrozenClassifier")

    def _freeze_encoder_and_segmentation(self):
        """Freeze encoder and segmentation parameters, keep classification trainable"""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.network.named_parameters():
            if 'classification_head' in name:
                param.requires_grad = True
                trainable_count += 1
                self.print_to_log_file(f"Trainable: {name}")
            else:
                param.requires_grad = False
                frozen_count += 1

        self.print_to_log_file(f"Frozen {frozen_count} encoder/seg parameters")
        self.print_to_log_file(f"Trainable {trainable_count} classification parameters")

    def _verify_parameter_setup(self):
        """Verify parameter freezing"""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        self.print_to_log_file("=== Stage 2 Parameter Setup ===")
        self.print_to_log_file(f"Total: {total_params:,}")
        self.print_to_log_file(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        self.print_to_log_file(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")

    def compute_loss(self, output, target, loss_dict=None):
        """Override to disable segmentation loss"""
        if loss_dict is None:
            loss_dict = {}

        # Set segmentation loss to zero
        loss_dict['seg_loss'] = torch.tensor(0.0, device=self.device)

        # Only compute classification loss
        cls_output = output['classification']
        cls_target = target['classification']
        cls_loss = self.classification_criterion(cls_output, cls_target)
        loss_dict['cls_loss'] = cls_loss * self.classification_weight

        # Total loss is just classification loss
        total_loss = loss_dict['cls_loss']
        loss_dict['total_loss'] = total_loss

        return total_loss

    def configure_optimizers(self):
        """Configure optimizer for only trainable (classification) parameters"""
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]

        self.print_to_log_file(f"Stage 2 optimizer configured for {len(trainable_params)} parameter groups")

        # Use lower learning rate for classification head fine-tuning
        classification_lr = self.initial_lr * 0.1  # 10x smaller LR

        optimizer = torch.optim.SGD(trainable_params, classification_lr,
                                   weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, self.num_epochs, power=0.9)

        self.print_to_log_file(f"Stage 2 learning rate: {classification_lr}")

        return optimizer, lr_scheduler
