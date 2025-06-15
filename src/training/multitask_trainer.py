import shutil
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import PolynomialLR
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
    with progressive training stages and uncertainty weighting
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        # Progressive training stages
        self.training_stages = ['enc_seg', 'enc_cls', 'joint_finetune', 'full']
        self.current_stage_idx = 0
        self.epochs_per_stage = [50, 25, 50, 100]  # Adjustable
        self.stage_epoch_counter = 0

        # Call parent constructor after to clear the stage
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Multi-task configuration
        self.multitask_config = self.configuration_manager.configuration.get('multitask_config', {
            'seg_weight': 1.0,
            'cls_weight': 0.5,
            'use_focal_loss': True,
            'focal_gamma': 2.0,
            'focal_alpha': 0.25
        })

        # Classification configuration
        self.num_classification_classes = 3  # subtype 0, 1, 2

        # Metrics tracking
        self.train_metrics = {'seg_loss': [], 'cls_loss': [], 'total_loss': []}
        self.val_metrics = {'seg_dice': [], 'cls_f1': [], 'pancreas_dice': [], 'lesion_dice': []}
        self.print_to_log_file(f"Stage schedule: {dict(zip(self.training_stages, self.epochs_per_stage))}")

    def set_custom_stage_epochs(self, custom_epochs_per_stage: List[int]):
        """
        Set custom epochs per stage for training
        """
        if len(custom_epochs_per_stage) != len(self.training_stages):
            raise ValueError(f"Custom epochs must match number of stages: {len(self.training_stages)}")
        self.epochs_per_stage = custom_epochs_per_stage
        self.print_to_log_file(f"Custom epochs per stage set: {self.epochs_per_stage}")

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
                classification_config=arch_init_kwargs.get('classification_config', {
                    'num_classes': 3,
                    'dropout_rate': 0.2,
                    'hidden_dims': [256, 128],
                    'use_all_features': True
                })
            )
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
                alpha=self.multitask_config.get('focal_alpha', 0.25),
                gamma=self.multitask_config.get('focal_gamma', 2.0)
            )
        else:
            self.cls_loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """Configure optimizer based on planner settings"""
        optimizer_config = self.configuration_manager.configuration.get('optimizer_config', {
            'optimizer': 'SGD',
            'initial_lr': 1e-2,
            'momentum': 0.99,
            'weight_decay': 3e-5
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

        lr_scheduler = PolynomialLR(optimizer, self.num_epochs, power=0.9)

        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        """Single training step with multi-task loss"""
        # Unpack batch - now returns 5-tuple: data, seg, seg_prev, properties
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

        # Compute losses
        loss_dict = self.network.compute_multitask_loss(
            output,
            {'segmentation': target_seg, 'classification': target_cls},
            self.seg_loss,
            self.cls_loss
        )

        total_loss = loss_dict['total_loss']

        # Backward pass
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

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': loss_dict['segmentation_loss'].detach().cpu().numpy(),
            'cls_loss': loss_dict['classification_loss'].detach().cpu().numpy(),
            'seg_weight': loss_dict['seg_weight'],
            'cls_weight': loss_dict['cls_weight']
        }

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
            'final_uncertainty_weights': {
                'seg_weight': torch.exp(-self.network.log_var_seg).item(),
                'cls_weight': torch.exp(-self.network.log_var_cls).item()
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
        """Validation step with multi-task metrics"""
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

            # Compute losses
            loss_dict = self.network.compute_multitask_loss(
                output,
                {'segmentation': target_seg, 'classification': target_cls},
                self.seg_loss,
                self.cls_loss
            )

            # Compute metrics
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

            # Uncertainty weights (current values)
            'uncertainty_weights': {
                'seg_weight': torch.exp(-self.network.log_var_seg).item(),
                'cls_weight': torch.exp(-self.network.log_var_cls).item(),
                'log_var_seg': self.network.log_var_seg.item(),
                'log_var_cls': self.network.log_var_cls.item()
            },

            # Training metrics history
            'train_metrics': getattr(self, 'train_metrics', {}),
            'val_metrics': getattr(self, 'val_metrics', {}),

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

    def load_checkpoint(self, filename: str) -> None:
        """
        Enhanced checkpoint loading with multi-task state restoration
        """
        checkpoint = torch.load(filename, map_location=self.device)

        # Load standard components
        self.network.load_state_dict(checkpoint['network_weights'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if self.lr_scheduler is not None and checkpoint['lr_scheduler_state'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

        if self.grad_scaler is not None and 'grad_scaler_state' in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

        # Restore multi-task specific state
        self.current_epoch = checkpoint['current_epoch']

        if 'current_stage_idx' in checkpoint:
            self.current_stage_idx = checkpoint['current_stage_idx']
            self.stage_epoch_counter = checkpoint.get('stage_epoch_counter', 0)

            # Restore training stage
            stage_name = checkpoint.get('training_stage', 'full')
            self.network.set_training_stage(stage_name)

            self.print_to_log_file(f"Restored training stage: {stage_name}")
            self.print_to_log_file(f"Stage epoch counter: {self.stage_epoch_counter}")

        # Restore metrics if available
        if 'train_metrics' in checkpoint:
            self.train_metrics = checkpoint['train_metrics']
        if 'val_metrics' in checkpoint:
            self.val_metrics = checkpoint['val_metrics']

        self.was_initialized = checkpoint.get('init', True)
        self.print_to_log_file(f"Loaded checkpoint: {filename}")


    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        loss_here = np.mean(outputs_collated['val_loss'])

        # Aggregate intersection/union then compute dice
        seg_dice = self._compute_aggregated_dice(outputs_collated, 'seg')
        pancreas_dice = self._compute_aggregated_dice(outputs_collated, 'pancreas')
        lesion_dice = self._compute_aggregated_dice(outputs_collated, 'lesion')

        cls_f1 = np.mean(outputs_collated['cls_f1'])

        # Print metrics for logging
        print(f"EPOCH {self.current_epoch}: val_loss={loss_here:.4f}, seg_dice={seg_dice:.4f}, pancreas_dice={pancreas_dice:.4f}, lesion_dice={lesion_dice:.4f}, cls_f1={cls_f1:.4f}")

        # Use pancreas dice for standard nnUNet logging
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

        # Check if we need to advance training stage
        if (self.current_epoch > 0 and
            self.stage_epoch_counter >= self.epochs_per_stage[self.current_stage_idx] and
            self.current_stage_idx < len(self.training_stages) - 1):

            # Save checkpoint before stage transition
            old_stage = self.training_stages[self.current_stage_idx]
            transition_checkpoint = f"checkpoint_before_{old_stage}_to_{self.training_stages[self.current_stage_idx + 1]}.pth"
            self.save_checkpoint(os.path.join(self.output_folder, transition_checkpoint))

            self.current_stage_idx += 1
            self.stage_epoch_counter = 0

            # Update network training stage
            stage_name = self.training_stages[self.current_stage_idx]
            self.network.set_training_stage(stage_name)

            self.print_to_log_file(f"Advanced to training stage: {stage_name}")
            self.print_to_log_file(f"Trainable parameters: {self.network.get_training_stage_info()['trainable_parameters']:,}")

            # Adjust learning rate for new stage
            if stage_name in ['enc_cls', 'joint_finetune']:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1  # Reduce LR for fine-tuning stages

        self.stage_epoch_counter += 1

    def on_epoch_end(self):
        """Log training progress and metrics"""
        super().on_epoch_end()

        # Log uncertainty weights
        if hasattr(self.network, 'log_var_seg'):
            seg_weight = torch.exp(-self.network.log_var_seg).item()
            cls_weight = torch.exp(-self.network.log_var_cls).item()

            self.print_to_log_file(f"seg_uncertainty_weight: {seg_weight}")
            self.print_to_log_file(f"cls_uncertainty_weight: {cls_weight}")

        # Periodic saving
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

    # def compute_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
    #     """Compute overall Dice score"""
    #     pred_binary = (pred.argmax(dim=1) > 0).float()
    #     target_binary = (target > 0).float()

    #     intersection = (pred_binary * target_binary).sum()
    #     union = pred_binary.sum() + target_binary.sum()

    #     dice = (2.0 * intersection) / (union + 1e-8)
    #     return dice.item()

    # def compute_pancreas_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
    #     """Compute pancreas-specific Dice score"""
    #     pred_pancreas = (pred.argmax(dim=1) > 0).float()  # Combined pancreas + lesion
    #     target_pancreas = (target > 0).float()

    #     intersection = (pred_pancreas * target_pancreas).sum()
    #     union = pred_pancreas.sum() + target_pancreas.sum()

    #     dice = (2.0 * intersection) / (union + 1e-8)
    #     return dice.item()

    # def compute_lesion_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
    #     """Compute lesion-specific Dice score"""
    #     pred_lesion = (pred.argmax(dim=1) == 2).float()  # Only lesion class
    #     target_lesion = (target == 2).float()

    #     intersection = (pred_lesion * target_lesion).sum()
    #     union = pred_lesion.sum() + target_lesion.sum()

    #     if union == 0:
    #         return 1.0  # Perfect score if no lesions present

    #     dice = (2.0 * intersection) / (union + 1e-8)
    #     return dice.item()

    def compute_macro_f1(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute macro-averaged F1 score for classification"""
        pred_classes = pred.argmax(dim=1).cpu().numpy()
        target_classes = target.cpu().numpy()

        f1_scores = []
        for class_idx in range(self.num_classification_classes):
            pred_class = (pred_classes == class_idx)
            target_class = (target_classes == class_idx)

            tp = (pred_class & target_class).sum()
            fp = (pred_class & ~target_class).sum()
            fn = (~pred_class & target_class).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            f1_scores.append(f1)

        return float(np.mean(f1_scores))

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

class FocalLoss(nn.Module):
    """Focal Loss for classification with class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


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