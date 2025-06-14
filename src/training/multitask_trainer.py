import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import PolynomialLR
import numpy as np
from typing import Union, Tuple, List, Dict, Any
import time

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, isdir, maybe_mkdir_p

from src.architectures.multitask_resenc_unet import MultiTaskResEncUNet
from src.training.dataloading.multitask_dataset import MultiTasknnUNetDataset


class nnUNetTrainerMultiTask(nnUNetTrainerNoDeepSupervision):
    """
    Multi-task trainer for pancreatic segmentation and classification
    with progressive training stages and uncertainty weighting
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Multi-task configuration
        self.multitask_config = self.configuration_manager.configuration.get('multitask_config', {
            'seg_weight': 1.0,
            'cls_weight': 0.5,
            'use_focal_loss': True,
            'focal_gamma': 2.0,
            'focal_alpha': 0.25
        })

        # Progressive training stages
        self.training_stages = ['enc_seg', 'enc_cls', 'joint_finetune', 'full']
        self.current_stage_idx = 0
        self.epochs_per_stage = [50, 25, 50, 100]  # Adjustable
        self.stage_epoch_counter = 0

        # Classification configuration
        self.num_classification_classes = 3  # subtype 0, 1, 2

        # Metrics tracking
        self.train_metrics = {'seg_loss': [], 'cls_loss': [], 'total_loss': []}
        self.val_metrics = {'seg_dice': [], 'cls_f1': [], 'pancreas_dice': [], 'lesion_dice': []}

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                 dataset_json: dict,
                                 configuration_manager: ConfigurationManager,
                                 num_input_channels: int,
                                 enable_deep_supervision: bool = False) -> nn.Module:
        """Build multi-task network"""

        num_stages = len(configuration_manager.configuration['conv_kernel_sizes'])
        dim = len(configuration_manager.configuration['conv_kernel_sizes'][0])
        conv_op = convert_dim_to_conv_op(dim)

        segmentation_network_class_name = configuration_manager.configuration['architecture']['network_class_name']
        architecture_kwargs = configuration_manager.configuration['architecture']['arch_kwargs']

        # Get classification config
        classification_config = configuration_manager.configuration['architecture'].get('classification_head', {
            'num_classes': 3,
            'dropout_rate': 0.2,
            'hidden_dims': [256, 128],
            'use_all_features': True
        })

        network = MultiTaskResEncUNet(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=architecture_kwargs['features_per_stage'],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.configuration['conv_kernel_sizes'],
            strides=configuration_manager.configuration['pool_op_kernel_sizes'],
            n_blocks_per_stage=architecture_kwargs['n_blocks_per_stage'],
            num_classes=plans_manager.get_label_manager(dataset_json).num_segmentation_classes,
            n_conv_per_stage_decoder=architecture_kwargs['n_conv_per_stage_decoder'],
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            classification_config=classification_config
        )

        return network

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
        # Unpack batch - now returns 5-tuple: data, seg, seg_prev, properties, classification
        data, target_seg, seg_prev, properties, target_cls = batch

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

    def validation_step(self, batch: dict) -> dict:
        """Validation step with multi-task metrics"""
        # Unpack batch
        data, target_seg, seg_prev, properties, target_cls = batch

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

            # Segmentation metrics
            seg_dice = self.compute_dice_score(seg_pred, target_seg)
            pancreas_dice = self.compute_pancreas_dice(seg_pred, target_seg)
            lesion_dice = self.compute_lesion_dice(seg_pred, target_seg)

            # Classification metrics
            cls_f1 = self.compute_macro_f1(cls_pred, target_cls)

        return {
            'val_loss': loss_dict['total_loss'].detach().cpu().numpy(),
            'seg_dice': seg_dice,
            'pancreas_dice': pancreas_dice,
            'lesion_dice': lesion_dice,
            'cls_f1': cls_f1
        }

    def on_epoch_start(self):
        """Handle training stage progression"""
        super().on_epoch_start()

        # Check if we need to advance training stage
        if (self.current_epoch > 0 and
            self.stage_epoch_counter >= self.epochs_per_stage[self.current_stage_idx] and
            self.current_stage_idx < len(self.training_stages) - 1):

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

            self.logger.log('train', 'seg_uncertainty_weight', seg_weight, self.current_epoch)
            self.logger.log('train', 'cls_uncertainty_weight', cls_weight, self.current_epoch)

    def compute_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute overall Dice score"""
        pred_binary = (pred.argmax(dim=1) > 0).float()
        target_binary = (target > 0).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()

        dice = (2.0 * intersection) / (union + 1e-8)
        return dice.item()

    def compute_pancreas_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute pancreas-specific Dice score"""
        pred_pancreas = (pred.argmax(dim=1) > 0).float()  # Combined pancreas + lesion
        target_pancreas = (target > 0).float()

        intersection = (pred_pancreas * target_pancreas).sum()
        union = pred_pancreas.sum() + target_pancreas.sum()

        dice = (2.0 * intersection) / (union + 1e-8)
        return dice.item()

    def compute_lesion_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute lesion-specific Dice score"""
        pred_lesion = (pred.argmax(dim=1) == 2).float()  # Only lesion class
        target_lesion = (target == 2).float()

        intersection = (pred_lesion * target_lesion).sum()
        union = pred_lesion.sum() + target_lesion.sum()

        if union == 0:
            return 1.0  # Perfect score if no lesions present

        dice = (2.0 * intersection) / (union + 1e-8)
        return dice.item()

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

    def _get_label_distribution(self, labels_dict: dict) -> dict:
        """Get distribution of classification labels"""
        from collections import Counter
        distribution = Counter(labels_dict.values())
        return {f'subtype_{i}': distribution.get(i, 0) for i in range(3)}


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