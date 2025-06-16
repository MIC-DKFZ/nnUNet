import shutil
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import PolynomialLR, CosineAnnealingWarmRestarts
import numpy as np
from typing import Union, Tuple, List, Dict, Any

# from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
# from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
# from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
# from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, isdir, maybe_mkdir_p
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.collate_outputs import collate_outputs

# from src.architectures.multitask_resenc_unet import MultiTaskResEncUNet
from src.training.dataloading.multitask_dataset import MultiTasknnUNetDataset, MultiTasknnUNetDataLoader


class nnUNetTrainerMultiTask(nnUNetTrainerNoDeepSupervision):
    """
    Multi-task trainer for pancreatic segmentation and classification
    with progressive training stages and manual weighting
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Progressive training stages
        self.training_stages = ['full']
        self.current_stage_idx = 0
        self.stage_epoch_counter = 0

        self.epochs_per_stage = [100]  # Max epochs per stage
        self.min_epochs_before_switch = 20  # Minimum epochs in full training
        self.switch_criteria = {
            'seg_dice_threshold': 0.95,
            'lesion_dice_threshold': 0.4,
            'stability_epochs': 3  # Number of consecutive epochs meeting criteria
        }
        self.criteria_met_count = 0  # Track consecutive epochs meeting criteria

        # Multi-task configuration with loss normalization
        self.multitask_config = self.configuration_manager.configuration.get('multitask_config', {
            'seg_weight': 1.0,
            'cls_weight': 0.5,
            'use_focal_loss': True,
            'focal_gamma': 2.0,
            'focal_alpha': [0.33, 0.33, 0.33],
            # Loss normalization settings
            'use_loss_normalization': True,
            'normalization_warmup_epochs': 10,
            'progressive_weighting': True,
            'ema_momentum': 0.1,
            'min_ema_value': 1e-6
        })

        # Loss magnitude normalization state (stored in trainer)
        self.loss_normalization = {
            'enabled': self.multitask_config.get('use_loss_normalization', True),
            'seg_loss_ema': None,
            'cls_loss_ema': None,
            'warmup_epochs': self.multitask_config.get('normalization_warmup_epochs', 10),
            'ema_momentum': self.multitask_config.get('ema_momentum', 0.1),
            'min_ema_value': self.multitask_config.get('min_ema_value', 1e-6),
            'initialized': False
        }


        # Classification configuration
        self.num_classification_classes = 3  # subtype 0, 1, 2

        # Pretrained weight path
        self.pretrained_checkpoint_path = plans.get('pretrained_checkpoint', None)

        # Metrics tracking
        self.train_metrics = {'seg_loss': [], 'cls_loss': [], 'total_loss': []}
        self.val_metrics = {'seg_dice': [], 'cls_f1': [], 'pancreas_dice': [], 'lesion_dice': []}
        self.print_to_log_file(f"Stage schedule: {dict(zip(self.training_stages, self.epochs_per_stage))}")

    def check_adaptive_switch(self, seg_dice, lesion_dice):
        """Check if we should adaptively switch training stages"""
        if (self.current_stage_idx == 0 and  # Currently in full training
            self.current_epoch >= self.min_epochs_before_switch and
            seg_dice >= self.switch_criteria['seg_dice_threshold'] and
            lesion_dice >= self.switch_criteria['lesion_dice_threshold']):

            self.criteria_met_count += 1
            self.print_to_log_file(f"Switch criteria met for {self.criteria_met_count} consecutive epochs")

            if self.criteria_met_count >= self.switch_criteria['stability_epochs']:
                return True
        else:
            self.criteria_met_count = 0  # Reset counter if criteria not met

        return False


    def set_custom_stage_epochs(self, custom_epochs_per_stage: List[int]):
        """
        Set custom epochs per stage for training
        """
        if len(custom_epochs_per_stage) != len(self.training_stages):
            raise ValueError(f"Custom epochs must match number of stages: {len(self.training_stages)}")
        self.epochs_per_stage = custom_epochs_per_stage
        self.num_epochs = sum(self.epochs_per_stage)
        self.print_to_log_file(f"Custom epochs per stage set: {dict(zip(self.training_stages, self.epochs_per_stage))}")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                    arch_init_kwargs: dict,
                                    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                    num_input_channels: int,
                                    num_output_channels: int,
                                    enable_deep_supervision: bool = True) -> nn.Module:
        """Build multi-task network"""
        # For now, create the network with the expected architecture
        from src.architectures.multitask_resenc_unet import MultiTaskResEncUNet

        # If this is our custom architecture, build it directly
        if architecture_class_name.split('.')[-1] == 'MultiTaskResEncUNet':
            # Convert string references to actual classes, ex: conv_op needs to be the class not str
            import importlib
            for key in arch_init_kwargs_req_import:
                if key in arch_init_kwargs and isinstance(arch_init_kwargs[key], str):
                    module_path, class_name = arch_init_kwargs[key].rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    arch_init_kwargs[key] = getattr(module, class_name)

            # We need to extract parameters from arch_init_kwargs
            network = MultiTaskResEncUNet(
                input_channels=num_input_channels,
                n_stages=arch_init_kwargs['n_stages'],
                features_per_stage=arch_init_kwargs['features_per_stage'],
                conv_op=arch_init_kwargs['conv_op'],
                kernel_sizes=arch_init_kwargs['kernel_sizes'],
                strides=arch_init_kwargs['strides'],
                n_blocks_per_stage=arch_init_kwargs['n_blocks_per_stage'],
                num_classes=num_output_channels,
                n_conv_per_stage_decoder=arch_init_kwargs.get('n_conv_per_stage_decoder'),
                conv_bias=arch_init_kwargs.get('conv_bias', True),
                norm_op=arch_init_kwargs.get('norm_op'),
                norm_op_kwargs=arch_init_kwargs.get('norm_op_kwargs'),
                dropout_op=arch_init_kwargs.get('dropout_op'),
                dropout_op_kwargs=arch_init_kwargs.get('dropout_op_kwargs'),
                nonlin=arch_init_kwargs.get('nonlin'),
                nonlin_kwargs=arch_init_kwargs.get('nonlin_kwargs'),
                deep_supervision=enable_deep_supervision,
                classification_config=arch_init_kwargs.get('classification_head', None)
            )

            # This follows the nnUNet pattern: network.apply(network.initialize)
            network.apply(network.initialize)
            # Post-initialization setup
            network.post_initialization_setup()
            return network
        else:
            raise ValueError(f"Wrong architecture: {architecture_class_name}")

    def _build_loss(self):
        """Build multi-task loss function"""
        # Segmentation loss (Dice + CE like standard nnUNet)
        dice_kwargs = {'batch_dice': self.configuration_manager.batch_dice,
                       'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}
        ce_kwargs = {}

        self.seg_loss = DC_and_CE_loss(
            dice_kwargs, ce_kwargs,
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        # Classification loss
        if self.multitask_config.get('use_focal_loss', True):
            self.cls_loss = FocalLoss(
                alpha=self.multitask_config.get('focal_alpha', [0.33, 0.33, 0.34]),
                gamma=self.multitask_config.get('focal_gamma', 2.0)
            )
        else:
            self.cls_loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """Configure optimizer based on planner settings"""
        optimizer_config = self.configuration_manager.configuration.get('optimizer_config', {
            'optimizer': 'Adam',
            'initial_lr': 1e-4,
            'momentum': 0.99,
            'weight_decay': 1e-5
        })

        if optimizer_config['optimizer'] == 'Adam':
            optimizer = Adam(
                self.network.parameters(),
                lr=optimizer_config['initial_lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:  # SGD
            optimizer = SGD(
                self.network.parameters(),
                lr=optimizer_config['initial_lr'],
                momentum=optimizer_config.get('momentum', 0.99),
                weight_decay=optimizer_config['weight_decay'],
                nesterov=True
            )

        lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=25,  # Restart every 25 epochs
                T_mult=1,
                eta_min=1e-6
            )
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        """Enhanced training step with loss normalization"""
        # Unpack batch
        data = batch['data']
        target_seg = batch['target']
        target_cls = batch['classification']
        keys = batch['keys']

        data = data.to(self.device, non_blocking=True)
        target_seg = target_seg.to(self.device, non_blocking=True)
        target_cls = torch.tensor(target_cls, dtype=torch.long).to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Forward pass
        output = self.network(data)

        # Compute losses with normalization (now handled in trainer)
        loss_dict = self.compute_multitask_loss_with_normalization(
            outputs=output,
            targets={'segmentation': target_seg, 'classification': target_cls}
        )

        total_loss = loss_dict['total_loss']

        # Backward pass (unchanged)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # Return enhanced metrics
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': loss_dict['segmentation_loss'].detach().cpu().numpy(),
            'cls_loss': loss_dict['classification_loss'].detach().cpu().numpy(),
            'seg_weight': loss_dict['seg_weight'],
            'cls_weight': loss_dict['cls_weight'],
            # Add normalization metrics if available
            **{k: v for k, v in loss_dict.items() if k.endswith('_ema') or k.endswith('_normalized') or k.endswith('_ratio')}
        }

    def check_initialization_health(self):
        """Debug method to verify initialization is working"""
        network = self.network.module if self.is_ddp else self.network

        if hasattr(network, 'attention_maps'):
            print("\n🔍 Initialization Health Check:")

            # Check manual weights
            print(f"  Manual weights: seg={network.seg_weight:.3f}, cls={network.cls_weight:.3f}")

            # Check attention module weights
            for name, module in network.named_modules():
                if 'spatial_attention' in name and isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    weight_std = module.weight.std().item()
                    print(f"  {name} weight std: {weight_std:.6f} {'✓' if 0.001 < weight_std < 0.1 else '⚠️'}")

            # Check classification head weights
            for name, module in network.named_modules():
                if 'classification_head' in name and isinstance(module, nn.Linear):
                    weight_std = module.weight.std().item()
                    expected_range = (0.01, 0.3) if 'classifier' not in name else (0.0001, 0.01)
                    in_range = expected_range[0] < weight_std < expected_range[1]
                    print(f"  {name} weight std: {weight_std:.6f} {'✓' if in_range else '⚠️'}")

    def on_train_start(self):
        """Override to add pretrained checkpoint loading after network initialization"""
        # Call parent's on_train_start (this initializes the network)
        super().on_train_start()

        # Synchronize network training stage with trainer's current stage
        current_stage_name = self.training_stages[self.current_stage_idx]
        self.network.set_training_stage(current_stage_name)
        self.print_to_log_file(f"Synchronized network training stage to: {current_stage_name}")

        # Set manual weights and loss normalization based on current stage
        if current_stage_name == 'enc_cls':
            self.loss_normalization['enabled'] = False
            self.network.set_manual_weights(0.0, self.multitask_config.get('cls_weight', 1.0))
            self.print_to_log_file(f"Stage {current_stage_name}: Disabled segmentation loss, enabled classification loss")
        elif current_stage_name == 'enc_seg':
            self.loss_normalization['enabled'] = False
            self.network.set_manual_weights(self.multitask_config.get('seg_weight', 1.0), 0.0)
            self.print_to_log_file(f"Stage {current_stage_name}: Enabled segmentation loss, disabled classification loss")
        else:
            # Full training or joint fine-tuning
            self.network.set_manual_weights(self.multitask_config.get('seg_weight', 1.0), self.multitask_config.get('cls_weight', 1.0))
            self.print_to_log_file(f"Stage {current_stage_name}: Both losses enabled")

        self.print_to_log_file(f"Set manual weights: seg={self.network.seg_weight}, cls={self.network.cls_weight}")

        # Check initialization after everything is set up
        if hasattr(self, 'check_initialization_health'):
            self.check_initialization_health()

        # After network is initialized, load pretrained weights if specified
        if hasattr(self, 'pretrained_checkpoint_path') and self.pretrained_checkpoint_path:
            if os.path.exists(self.pretrained_checkpoint_path):
                self.print_to_log_file(f"Loading pretrained checkpoint: {self.pretrained_checkpoint_path}")
                self._load_checkpoint(self.pretrained_checkpoint_path)
            else:
                self.print_to_log_file(f"Warning: Pretrained checkpoint not found: {self.pretrained_checkpoint_path}")

    def set_pretrained_checkpoint(self, checkpoint_path: str):
        """Set the path to pretrained checkpoint to load after network initialization"""
        self.pretrained_checkpoint_path = checkpoint_path
        self.print_to_log_file(f"Pretrained checkpoint path set: {checkpoint_path}")

    def on_train_end(self):
        """Save final checkpoint with complete training information"""
        super().on_train_end()

        # Save final checkpoint
        final_checkpoint_path = os.path.join(self.output_folder, "checkpoint_final.pth")
        self.save_checkpoint(final_checkpoint_path)

        # Save a summary of the training process
        training_summary = {
            'total_epochs': self.current_epoch + 1,
            'training_stages_completed': self.training_stages[:self.current_stage_idx + 1],
            'epochs_per_stage_actual': self._get_actual_epochs_per_stage(),
            'final_manual_weights': {
                'seg_weight': self.network.seg_weight,
                'cls_weight': self.network.cls_weight
            },
            'final_stage_info': self.network.get_training_stage_info()
        }

        import json
        summary_path = os.path.join(self.output_folder, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)

        self.print_to_log_file("Training completed. Final checkpoint and summary saved.")

    def _get_actual_epochs_per_stage(self):
        """Helper to track actual epochs spent in each stage"""
        # This would need to be tracked during training
        # For now, return the planned epochs
        return dict(zip(self.training_stages, self.epochs_per_stage))

    def validation_step(self, batch: dict) -> dict:
        """Enhanced validation step"""
        # Unpack batch
        data = batch['data']
        target_seg = batch['target']
        target_cls = batch['classification']
        keys = batch['keys']

        data = data.to(self.device, non_blocking=True)
        target_seg = target_seg.to(self.device, non_blocking=True)
        target_cls = torch.tensor(target_cls, dtype=torch.long).to(self.device, non_blocking=True)

        with torch.no_grad():
            output = self.network(data)

            # Compute losses (no EMA updates during validation)
            loss_dict = self.compute_multitask_loss_with_normalization(
                outputs=output,
                targets={'segmentation': target_seg, 'classification': target_cls}
            )

            # Compute metrics (your existing metric computation)
            seg_pred = torch.softmax(output['segmentation'], dim=1)
            cls_pred = torch.softmax(output['classification'], dim=1)

            # Compute raw intersection/union for dice metrics
            seg_intersect, seg_union = self.compute_dice_components(seg_pred, target_seg, mode='overall')
            pancreas_intersect, pancreas_union = self.compute_dice_components(seg_pred, target_seg, mode='pancreas')
            lesion_intersect, lesion_union = self.compute_dice_components(seg_pred, target_seg, mode='lesion')

            # Classification metrics
            cls_f1 = self.compute_macro_f1(cls_pred, target_cls)

        return {
            'val_loss': loss_dict['total_loss'].detach().cpu().numpy(),
            'seg_intersect': seg_intersect,
            'seg_union': seg_union,
            'pancreas_intersect': pancreas_intersect,
            'pancreas_union': pancreas_union,
            'lesion_intersect': lesion_intersect,
            'lesion_union': lesion_union,
            'cls_f1': cls_f1
        }

    def save_checkpoint(self, filename: str) -> None:
        """
        Enhanced checkpoint saving with multi-task specific information
        """
        checkpoint = {
            'network_weights': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lr_scheduler_state': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'epoch': self.current_epoch,
            'current_epoch': self.current_epoch,
            'init': self.was_initialized,

            # Multi-task specific state
            'training_stage': self.training_stages[self.current_stage_idx],
            'current_stage_idx': self.current_stage_idx,
            'stage_epoch_counter': self.stage_epoch_counter,
            'epochs_per_stage': self.epochs_per_stage,

            # Manual weights (current values)
            'manual_weights': {
                'seg_weight': self.network.seg_weight,
                'cls_weight': self.network.cls_weight
            },

            # Training metrics history
            'train_metrics': getattr(self, 'train_metrics', {}),
            'val_metrics': getattr(self, 'val_metrics', {}),

            # Logger state to prevent IndexError when switching strategies
            'logger_state': self.logger.get_checkpoint() if hasattr(self, 'logger') and self.logger is not None else None,

            # Loss normalization state
            'loss_normalization': self.loss_normalization,

            # Additional metadata
            'multitask_config': self.multitask_config,
            'training_stage_info': self.network.get_training_stage_info()
        }

        # Add grad scaler state if using mixed precision
        if self.grad_scaler is not None:
            checkpoint['grad_scaler_state'] = self.grad_scaler.state_dict()

        torch.save(checkpoint, filename)
        self.print_to_log_file(f"Saved checkpoint: {filename}")
        self.print_to_log_file(f"Stage: {checkpoint['training_stage']}, Stage Epoch: {self.stage_epoch_counter}")

    def _load_checkpoint(self, filename: str) -> None:
        """Enhanced checkpoint loading with logger state management"""
        checkpoint = torch.load(filename, map_location=self.device)

        # Load standard components
        self.network.load_state_dict(checkpoint['network_weights'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if self.lr_scheduler is not None and checkpoint['lr_scheduler_state'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

        if self.grad_scaler is not None and 'grad_scaler_state' in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

        # Restore multi-task specific state
        # self.current_epoch = checkpoint['current_epoch']

        if 'current_stage_idx' in checkpoint:
            self.current_stage_idx = checkpoint['current_stage_idx']
            self.stage_epoch_counter = checkpoint.get('stage_epoch_counter', 0) - 1

            # Restore training stage
            stage_name = checkpoint.get('training_stage', 'full')
            self.network.set_training_stage(stage_name)

            self.print_to_log_file(f"Restored training stage: {stage_name}")
            self.print_to_log_file(f"Stage epoch counter: {self.stage_epoch_counter}")

        # Restore loss normalization state
        if 'loss_normalization' in checkpoint:
            self.loss_normalization.update(checkpoint['loss_normalization'])
            self.print_to_log_file(f"Restored loss normalization state: {self.loss_normalization}")

        # Restore metrics if available
        if 'train_metrics' in checkpoint:
            self.train_metrics = checkpoint['train_metrics']
        if 'val_metrics' in checkpoint:
            self.val_metrics = checkpoint['val_metrics']

        self.was_initialized = checkpoint.get('init', True)
        self.print_to_log_file(f"Loaded checkpoint: {filename}")

    def _handle_logger_state_on_checkpoint_load(self, checkpoint: dict) -> None:
        """
        Handle logger state restoration or reset when loading checkpoint.
        This prevents IndexError when switching training strategies.
        """
        if 'logger_state' in checkpoint and checkpoint['logger_state'] is not None:
            # Try to restore logger state if available
            try:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.load_checkpoint(checkpoint['logger_state'])
                    self.print_to_log_file("✓ Restored logger state from checkpoint")
                else:
                    self.print_to_log_file("⚠️ Logger not initialized, cannot restore logger state")
            except Exception as e:
                self.print_to_log_file(f"⚠️ Failed to restore logger state: {e}")
                self._reset_logger_for_fresh_start()
        else:
            # No logger state in checkpoint or switching strategies - reset logger
            self.print_to_log_file("No logger state in checkpoint - resetting logger for fresh start")
            self._reset_logger_for_fresh_start()

    def _reset_logger_for_fresh_start(self) -> None:
        """
        Reset logger arrays to start fresh training.
        This prevents IndexError when switching strategies or starting new training.
        """
        if hasattr(self, 'logger') and self.logger is not None:
            # Reset all logging arrays to empty lists
            self.logger.my_fantastic_logging = {
                'mean_fg_dice': list(),
                'ema_fg_dice': list(),
                'dice_per_class_or_region': list(),
                'train_losses': list(),
                'val_losses': list(),
                'lrs': list(),
                'epoch_start_timestamps': list(),
                'epoch_end_timestamps': list()
            }

            # Reset epoch counters to align with logger
            self.current_epoch = 0
            self.stage_epoch_counter = 0

            self.print_to_log_file("✓ Reset logger arrays and epoch counters for fresh start")
            self.print_to_log_file(f"Current epoch reset to: {self.current_epoch}")
            self.print_to_log_file(f"Stage epoch counter reset to: {self.stage_epoch_counter}")
        else:
            self.print_to_log_file("⚠️ Logger not available for reset")

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        loss_here = np.mean(outputs_collated['val_loss'])

        # Aggregate intersection/union then compute dice
        seg_dice = self._compute_aggregated_dice(outputs_collated, 'seg')
        pancreas_dice = self._compute_aggregated_dice(outputs_collated, 'pancreas')
        lesion_dice = self._compute_aggregated_dice(outputs_collated, 'lesion')

        cls_f1 = np.mean(outputs_collated['cls_f1'])

        # Print metrics for logging
        self.print_to_log_file(f"EPOCH {self.current_epoch}: val_loss={loss_here:.4f}, seg_dice={seg_dice:.4f}, pancreas_dice={pancreas_dice:.4f}, lesion_dice={lesion_dice:.4f}, cls_f1={cls_f1:.4f}")
        # print(f"Running epoch count across stages: {self.current_epoch}")
        # print(f"Current epoch at stage {self.training_stages[self.current_stage_idx]}: {self.stage_epoch_counter}")
        # Use pancreas dice for standard nnUNet logging
        # self.print_to_log_file(self.logger.my_fantastic_logging['val_losses'])
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('mean_fg_dice', pancreas_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', [lesion_dice], self.current_epoch)

    def compute_dice_components(self, pred: torch.Tensor, target: torch.Tensor, mode: str):
        """Compute intersection and union for dice calculation"""
        if mode == 'overall':
            pred_binary = (pred.argmax(dim=1) > 0).float()
            target_binary = (target > 0).float()
        elif mode == 'pancreas':
            pred_binary = (pred.argmax(dim=1) > 0).float()  # Same as overall for pancreas+lesion
            target_binary = (target > 0).float()
        elif mode == 'lesion':
            pred_binary = (pred.argmax(dim=1) == 2).float()
            target_binary = (target == 2).float()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        intersection = (pred_binary * target_binary).sum().item()
        union = (pred_binary.sum() + target_binary.sum()).item()

        return intersection, union

    def _compute_aggregated_dice(self, outputs_collated: dict, prefix: str) -> float:
        """Compute dice from aggregated intersection/union"""
        total_intersection = np.sum(outputs_collated[f'{prefix}_intersect'])
        total_union = np.sum(outputs_collated[f'{prefix}_union'])

        if total_union == 0:
            return 1.0

        dice = (2.0 * total_intersection) / (total_union + 1e-8)
        return min(dice, 1.0)  # Clamp to max 1.0

    def on_epoch_start(self):
        """Handle training stage progression"""
        super().on_epoch_start()

        # Increment stage epoch counter first
        self.stage_epoch_counter += 1

        # Check if we need to advance training stage
        if (self.current_epoch > 0 and
            self.stage_epoch_counter >= self.epochs_per_stage[self.current_stage_idx] and
            self.current_stage_idx < len(self.training_stages) - 1):

            # Save checkpoint before stage transition
            old_stage = self.training_stages[self.current_stage_idx]
            transition_checkpoint = f"checkpoint_before_{old_stage}_to_{self.training_stages[self.current_stage_idx + 1]}.pth"
            self.save_checkpoint(os.path.join(self.output_folder, transition_checkpoint))

            self.current_stage_idx += 1
            self.stage_epoch_counter = 1  # Reset to 1 since we're starting the new stage

            # Update network training stage and synchronize loss settings
            stage_name = self.training_stages[self.current_stage_idx]
            self.network.set_training_stage(stage_name)

            # Update loss normalization and manual weights for new stage
            if stage_name == 'enc_cls':
                self.loss_normalization['enabled'] = False
                self.network.set_manual_weights(0.0, self.multitask_config.get('cls_weight', 1.0))
                self.print_to_log_file(f"Stage {stage_name}: Disabled segmentation loss, enabled classification loss")
            elif stage_name == 'enc_seg':
                self.loss_normalization['enabled'] = False
                self.network.set_manual_weights(self.multitask_config.get('seg_weight', 1.0), 0.0)
                self.print_to_log_file(f"Stage {stage_name}: Enabled segmentation loss, disabled classification loss")
            else:
                # Re-enable loss normalization for full training stages
                self.loss_normalization['enabled'] = self.multitask_config.get('use_loss_normalization', True)
                self.network.set_manual_weights(self.multitask_config.get('seg_weight', 1.0), self.multitask_config.get('cls_weight', 1.0))
                self.print_to_log_file(f"Stage {stage_name}: Both losses enabled, normalization: {self.loss_normalization['enabled']}")

            self.print_to_log_file(f"Advanced to training stage: {stage_name} (from {old_stage})")
            self.print_to_log_file(f"Stage epoch counter reset to: {self.stage_epoch_counter}")
            self.print_to_log_file(f"Manual weights updated: seg={self.network.seg_weight}, cls={self.network.cls_weight}")
            self.print_to_log_file(f"Trainable parameters: {self.network.get_training_stage_info()['trainable_parameters']:,}")

            # Adjust learning rate for new stage
            if stage_name in ['enc_cls', 'joint_finetune']:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1  # Reduce LR for fine-tuning stages

        # Log current stage info
        current_stage_name = self.training_stages[self.current_stage_idx]
        self.print_to_log_file(f"Epoch {self.current_epoch}: Stage '{current_stage_name}', Stage Epoch {self.stage_epoch_counter}/{self.epochs_per_stage[self.current_stage_idx]}")

        # Log loss disabling status for debugging
        if current_stage_name in ['enc_seg', 'enc_cls']:
            self.print_to_log_file(f"Loss disabling active - seg_weight: {self.network.seg_weight}, cls_weight: {self.network.cls_weight}")

    def on_epoch_end(self):
        """Enhanced epoch end with normalization logging"""
        super().on_epoch_end()

        # Log loss normalization status every 10 epochs
        if self.current_epoch % 10 == 0 and self.loss_normalization['initialized']:
            seg_ema = self.loss_normalization['seg_loss_ema']
            cls_ema = self.loss_normalization['cls_loss_ema']
            ratio = cls_ema / seg_ema if seg_ema > 0 else 1.0

            self.print_to_log_file(
                f"Loss Normalization - Epoch {self.current_epoch}: "
                f"Seg EMA: {seg_ema:.4f}, Cls EMA: {cls_ema:.4f}, Ratio: {ratio:.2f}"
            )

        # Log manual weights
        if hasattr(self.network, 'seg_weight'):
            self.print_to_log_file(f"seg_manual_weight: {self.network.seg_weight}")
            self.print_to_log_file(f"cls_manual_weight: {self.network.cls_weight}")

        # Periodic saving (your existing logic)
        if hasattr(self, 'save_every') and self.save_every is not None:
            if (self.current_epoch + 1) % self.save_every == 0:
                checkpoint_name = f"checkpoint_epoch_{self.current_epoch + 1}.pth"
                checkpoint_path = os.path.join(self.output_folder, checkpoint_name)
                self.save_checkpoint(checkpoint_path)

        # Save at end of each training stage
        stage_name = self.training_stages[self.current_stage_idx]
        if self.stage_epoch_counter == self.epochs_per_stage[self.current_stage_idx]:
            stage_checkpoint_name = f"checkpoint_end_of_{stage_name}.pth"
            stage_checkpoint_path = os.path.join(self.output_folder, stage_checkpoint_name)
            self.save_checkpoint(stage_checkpoint_path)
            self.print_to_log_file(f"Saved end-of-stage checkpoint: {stage_checkpoint_name}")
            self.stage_epoch_counter = 0  # Reset counter for next stage

    def should_switch_to_cls_focus(self, seg_dice, lesion_dice):
        """Check if we should switch from full training to classification focus"""
        return seg_dice >= 0.95 and lesion_dice >= 0.4

    def compute_macro_f1(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute macro-averaged F1 score for classification with diagnostics"""
        pred_classes = pred.argmax(dim=1).cpu().numpy()
        target_classes = target.cpu().numpy()

        # Compute detailed metrics
        f1_scores = []
        detailed_metrics = []

        for class_idx in range(self.num_classification_classes):
            pred_class = (pred_classes == class_idx)
            target_class = (target_classes == class_idx)

            tp = (pred_class & target_class).sum()
            fp = (pred_class & ~target_class).sum()
            fn = (~pred_class & target_class).sum()
            tn = (~pred_class & ~target_class).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            f1_scores.append(f1)
            detailed_metrics.append({
                'class': class_idx,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': target_class.sum()  # Number of true instances
            })

        macro_f1 = float(np.mean(f1_scores))
        return macro_f1

    def additional_classification_diagnostics(self, pred: torch.Tensor, target: torch.Tensor):
        """Additional diagnostic function to call during validation"""
        pred_probs = torch.softmax(pred, dim=1).cpu().numpy()
        pred_classes = pred.argmax(dim=1).cpu().numpy()
        target_classes = target.cpu().numpy()

        # Check prediction confidence distribution
        max_probs = pred_probs.max(axis=1)

        # Check if model is stuck in local minima
        pred_variance = np.var(pred_probs, axis=0)  # Variance across samples for each class

        diagnostics = {
            'prediction_entropy': -np.sum(pred_probs * np.log(pred_probs + 1e-8), axis=1).mean(),
            'max_confidence': max_probs.mean(),
            'min_confidence': max_probs.min(),
            'prediction_variance_per_class': pred_variance,
            'unique_predictions': len(np.unique(pred_classes)),
            'prediction_distribution': np.bincount(pred_classes, minlength=self.num_classification_classes),
            'target_distribution': np.bincount(target_classes, minlength=self.num_classification_classes)
        }

        return diagnostics

    def get_tr_and_val_datasets(self):
        """Get training and validation datasets with multi-task support"""
        # Get case identifiers
        tr_keys, val_keys = self.do_split()

        # Create multi-task datasets
        dataset_tr = MultiTasknnUNetDataset(
            folder=self.preprocessed_dataset_folder,
            identifiers=tr_keys,
            folder_with_segs_from_previous_stage=None
        )

        dataset_val = MultiTasknnUNetDataset(
            folder=self.preprocessed_dataset_folder,
            identifiers=val_keys,
            folder_with_segs_from_previous_stage=None
        )

        # Log classification distribution
        tr_dist = self._get_label_distribution(dataset_tr.classification_labels)
        val_dist = self._get_label_distribution(dataset_val.classification_labels)

        self.print_to_log_file(f"Training classification distribution: {tr_dist}")
        self.print_to_log_file(f"Validation classification distribution: {val_dist}")

        return dataset_tr, dataset_val

    def get_dataloaders(self):
        # Get patch size and deep supervision scales
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        # Configure data augmentation
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes) = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # Get transforms
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )

        # Use your custom datasets
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = MultiTasknnUNetDataLoader(
            dataset_tr, self.batch_size, initial_patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        dl_val = MultiTasknnUNetDataLoader(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size, self.configuration_manager.patch_size,
            self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        # Standard nnUNet augmenter setup
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002
            )

        # Initialize dataloaders
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)

        return mt_gen_train, mt_gen_val

    def _get_label_distribution(self, labels_dict: dict) -> dict:
        """Get distribution of classification labels"""
        from collections import Counter
        distribution = Counter(labels_dict.values())
        return {f'subtype_{i}': distribution.get(i, 0) for i in range(3)}

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if self.dataloader_val is None:
            _, self.dataloader_val = self.get_dataloaders()

    def run_validation(self):
        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):
                val_outputs.append(self.validation_step(next(self.dataloader_val)))
            self.on_validation_epoch_end(val_outputs)

        self.on_epoch_end()

    def initialize_loss_normalization(self, seg_loss_value: float, cls_loss_value: float):
        """Initialize the EMAs with first computed losses"""
        if self.loss_normalization['enabled'] and not self.loss_normalization['initialized']:
            self.loss_normalization['seg_loss_ema'] = seg_loss_value
            self.loss_normalization['cls_loss_ema'] = cls_loss_value
            self.loss_normalization['initialized'] = True

            self.print_to_log_file(f"✓ Initialized loss EMAs - Seg: {seg_loss_value:.4f}, Cls: {cls_loss_value:.4f}")

    def update_loss_emas(self, seg_loss_value: float, cls_loss_value: float):
        """Update exponential moving averages of loss magnitudes"""
        if not self.loss_normalization['enabled'] or not self.loss_normalization['initialized']:
            return

        # Don't update EMAs too early
        if self.current_epoch < 5:  # Start updating from epoch 5
            return

        # Determine momentum based on training stage
        if self.current_epoch < self.loss_normalization['warmup_epochs']:
            momentum = 0.3  # Higher momentum = faster adaptation during warmup
        else:
            momentum = self.loss_normalization['ema_momentum']  # Stable momentum

        # Update EMAs
        self.loss_normalization['seg_loss_ema'] = (
            (1 - momentum) * self.loss_normalization['seg_loss_ema'] +
            momentum * seg_loss_value
        )
        self.loss_normalization['cls_loss_ema'] = (
            (1 - momentum) * self.loss_normalization['cls_loss_ema'] +
            momentum * cls_loss_value
        )

        # Ensure minimum values to prevent numerical issues
        min_val = self.loss_normalization['min_ema_value']
        self.loss_normalization['seg_loss_ema'] = max(self.loss_normalization['seg_loss_ema'], min_val)
        self.loss_normalization['cls_loss_ema'] = max(self.loss_normalization['cls_loss_ema'], min_val)

    def compute_multitask_loss_with_normalization(self, outputs: dict, targets: dict) -> dict:
        """Compute multi-task loss with optional magnitude normalization"""
        seg_pred = outputs['segmentation']
        cls_pred = outputs['classification']
        seg_target = targets['segmentation']
        cls_target = targets['classification']

        # Compute individual losses
        seg_loss = self.seg_loss(seg_pred, seg_target)
        cls_loss = self.cls_loss(cls_pred, cls_target)

        # Convert to float values for EMA tracking
        seg_loss_value = seg_loss.item()
        cls_loss_value = cls_loss.item()

        # Initialize EMAs on first call
        if not self.loss_normalization['initialized']:
            self.initialize_loss_normalization(seg_loss_value, cls_loss_value)

        # Update EMAs
        self.update_loss_emas(seg_loss_value, cls_loss_value)

        # Stage-specific loss handling - use network's training stage for consistency
        current_stage = self.network.training_stage

        if current_stage == 'enc_seg':
            # Only segmentation loss - classification loss is disabled
            if self.current_epoch % 5 == 0:  # Log every 5 epochs to avoid spam
                self.print_to_log_file(f"Stage {current_stage}: Using only segmentation loss (cls disabled) - seg_loss: {seg_loss_value:.4f}, cls_loss: {cls_loss_value:.4f}")
            return self._create_loss_dict(
                total_loss=seg_loss,
                seg_loss=seg_loss,
                cls_loss=cls_loss,
                seg_weight=1.0,
                cls_weight=0.0,
                seg_loss_raw=seg_loss_value,
                cls_loss_raw=cls_loss_value
            )
        elif current_stage == 'enc_cls':
            # Only classification loss - segmentation loss is disabled
            if self.current_epoch % 5 == 0:  # Log every 5 epochs to avoid spam
                self.print_to_log_file(f"Stage {current_stage}: Using only classification loss (seg disabled) - seg_loss: {seg_loss_value:.4f}, cls_loss: {cls_loss_value:.4f}")
            return self._create_loss_dict(
                total_loss=cls_loss,
                seg_loss=seg_loss,
                cls_loss=cls_loss,
                seg_weight=0.0,
                cls_weight=1.0,
                seg_loss_raw=seg_loss_value,
                cls_loss_raw=cls_loss_value
            )
        else:
            # Multi-task loss computation (full training or joint fine-tuning)
            if self._should_use_normalization():
                return self._compute_normalized_multitask_loss(seg_loss, cls_loss, seg_loss_value, cls_loss_value)
            else:
                return self._compute_manual_weighted_loss(seg_loss, cls_loss, seg_loss_value, cls_loss_value)

    def _should_use_normalization(self) -> bool:
        """Check if we should use loss normalization"""
        return (
            self.loss_normalization['enabled'] and
            self.loss_normalization['initialized'] and
            self.current_epoch >= 5  # Start normalization from epoch 5
        )

    def _create_loss_dict(self, total_loss, seg_loss, cls_loss, seg_weight, cls_weight,
                         seg_loss_raw, cls_loss_raw, seg_loss_norm=None, cls_loss_norm=None,
                         seg_ema=None, cls_ema=None):
        """Create standardized loss dictionary"""
        loss_dict = {
            'total_loss': total_loss,
            'segmentation_loss': seg_loss,
            'classification_loss': cls_loss,
            'seg_weight': seg_weight,
            'cls_weight': cls_weight,
            'seg_loss_raw': seg_loss_raw,
            'cls_loss_raw': cls_loss_raw
        }

        # Add normalization info if available
        if seg_loss_norm is not None:
            loss_dict.update({
                'seg_loss_normalized': seg_loss_norm,
                'cls_loss_normalized': cls_loss_norm,
                'seg_loss_ema': seg_ema,
                'cls_loss_ema': cls_ema,
                'ema_ratio': cls_ema / seg_ema if seg_ema > 0 else 1.0
            })

        return loss_dict

    def _get_current_weights(self) -> Tuple[float, float]:
        """Get current loss weights (progressive or fixed)"""
        if self.multitask_config.get('progressive_weighting', True):
            # Progressive weighting schedule
            if self.current_epoch < 50:
                return 0.7, 0.3  # Early: slight segmentation focus
            elif self.current_epoch < 150:
                return 0.6, 0.4  # Mid: balanced
            else:
                return 0.5, 0.5  # Late: equal weight
        else:
            # Fixed weights from config
            return (
                self.multitask_config.get('seg_weight', 1.0),
                self.multitask_config.get('cls_weight', 0.5)
            )

    def _compute_normalized_multitask_loss(self, seg_loss, cls_loss, seg_loss_value, cls_loss_value):
        """Compute loss with magnitude normalization"""
        seg_ema = self.loss_normalization['seg_loss_ema']
        cls_ema = self.loss_normalization['cls_loss_ema']

        # Normalize losses by their typical magnitudes
        seg_loss_normalized = seg_loss / seg_ema
        cls_loss_normalized = cls_loss / cls_ema

        # Get weights (progressive or fixed)
        seg_weight, cls_weight = self._get_current_weights()

        # Compute normalized total loss
        total_loss = seg_weight * seg_loss_normalized + cls_weight * cls_loss_normalized

        return self._create_loss_dict(
            total_loss=total_loss,
            seg_loss=seg_loss,
            cls_loss=cls_loss,
            seg_weight=seg_weight,
            cls_weight=cls_weight,
            seg_loss_raw=seg_loss_value,
            cls_loss_raw=cls_loss_value,
            seg_loss_norm=seg_loss_normalized.item(),
            cls_loss_norm=cls_loss_normalized.item(),
            seg_ema=seg_ema,
            cls_ema=cls_ema
        )

    def _compute_manual_weighted_loss(self, seg_loss, cls_loss, seg_loss_value, cls_loss_value):
        """Fallback to manual weighting"""
        seg_weight = self.multitask_config.get('seg_weight', 1.0)
        cls_weight = self.multitask_config.get('cls_weight', 0.5)

        total_loss = seg_weight * seg_loss + cls_weight * cls_loss

        return self._create_loss_dict(
            total_loss=total_loss,
            seg_loss=seg_loss,
            cls_loss=cls_loss,
            seg_weight=seg_weight,
            cls_weight=cls_weight,
            seg_loss_raw=seg_loss_value,
            cls_loss_raw=cls_loss_value
        )

    def get_loss_normalization_status(self) -> dict:
        """Get current status of loss normalization for debugging"""
        return {
            'enabled': self.loss_normalization['enabled'],
            'initialized': self.loss_normalization['initialized'],
            'current_epoch': self.current_epoch,
            'seg_loss_ema': self.loss_normalization['seg_loss_ema'],
            'cls_loss_ema': self.loss_normalization['cls_loss_ema'],
            'ema_ratio': (
                self.loss_normalization['cls_loss_ema'] / self.loss_normalization['seg_loss_ema']
                if self.loss_normalization['seg_loss_ema'] and self.loss_normalization['cls_loss_ema']
                else None
            ),
            'warmup_epochs': self.loss_normalization['warmup_epochs'],
            'should_use_normalization': self._should_use_normalization()
        }
class FocalLoss(nn.Module):
    """
    Improved Focal Loss for multi-class classification with class imbalance
    Supports both single alpha and per-class alpha weighting
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Handle different alpha configurations
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t
        pt = torch.exp(-ce_loss)

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)

            if len(self.alpha) > 1:  # Per-class alpha
                alpha_t = self.alpha[targets]
            else:  # Single alpha for all classes
                alpha_t = self.alpha[0]
        else:
            alpha_t = 1.0

        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def convert_dim_to_conv_op(dimension: int):
    """Convert dimension to conv operation"""
    if dimension == 1:
        return nn.Conv1d
    elif dimension == 2:
        return nn.Conv2d
    elif dimension == 3:
        return nn.Conv3d
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")


def get_matching_instancenorm(conv_op):
    """Get matching instance normalization for conv operation"""
    if conv_op == nn.Conv1d:
        return nn.InstanceNorm1d
    elif conv_op == nn.Conv2d:
        return nn.InstanceNorm2d
    elif conv_op == nn.Conv3d:
        return nn.InstanceNorm3d
    else:
        raise ValueError(f"No matching InstanceNorm for {conv_op}")
