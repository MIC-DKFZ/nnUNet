import torch
from torch import nn
import numpy as np
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from src.architectures.MultiTaskResEncUNet import MultiTaskResEncUNet, MultiTaskChannelAttentionResEncUNet, MultiTaskEfficientAttentionResEncUNet
from src.losses.multitask_losses import MultiTaskLoss
# from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
# from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op
# from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainerMultiTask(nnUNetTrainerNoDeepSupervision):
    """Multi-task trainer for segmentation + classification"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Multi-task specific parameters
        self.num_classification_classes = 3  # Update based on your subtypes
        self.seg_weight = 1.0
        self.cls_weight = 0.5
        self.loss_type = 'dice_ce'  # Options: 'dice_ce', 'focal', 'tversky'

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build the multi-task network architecture.
        This method follows the nnUNetv2 trainer interface but builds our custom multi-task network.
        """

        # Extract network configuration from arch_init_kwargs
        n_stages = arch_init_kwargs.get('n_stages', None)
        features_per_stage = arch_init_kwargs.get('features_per_stage', None)
        conv_op = arch_init_kwargs.get('conv_op', 'torch.nn.Conv3d')
        kernel_sizes = arch_init_kwargs.get('kernel_sizes', None)
        strides = arch_init_kwargs.get('strides', None)
        n_blocks_per_stage = arch_init_kwargs.get('n_blocks_per_stage', None)
        n_conv_per_stage_decoder = arch_init_kwargs.get('n_conv_per_stage_decoder', None)

        # Import conv_op if it's a string
        if isinstance(conv_op, str):
            import pydoc
            conv_op = pydoc.locate(conv_op)

        # Determine conv_or_blocks_per_stage for ResidualEncoderUNet
        if 'ResidualEncoderUNet' in architecture_class_name:
            conv_or_blocks_per_stage = {
                'n_blocks_per_stage': n_blocks_per_stage,
                'n_conv_per_stage_decoder': n_conv_per_stage_decoder
            }
        else:
            conv_or_blocks_per_stage = {
                'n_conv_per_stage': n_blocks_per_stage,
                'n_conv_per_stage_decoder': n_conv_per_stage_decoder
            }

        # Select the architecture class based on the plan configuration or argument
        if architecture_class_name == "MultiTaskResEncUNet":
            net_cls = MultiTaskResEncUNet
        elif architecture_class_name == "MultiTaskChannelAttentionResEncUNet":
            net_cls = MultiTaskChannelAttentionResEncUNet
        elif architecture_class_name == "MultiTaskEfficientAttentionResEncUNet":
            net_cls = MultiTaskEfficientAttentionResEncUNet
        else:
            raise ValueError(f"Unknown architecture_class_name: {architecture_class_name}")

        # Build the multi-task network
        network = net_cls(
            input_channels=num_input_channels,
            num_segmentation_classes=num_output_channels,
            num_classification_classes=arch_init_kwargs.get('num_classification_classes', 3),
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            **conv_or_blocks_per_stage,
            conv_bias=arch_init_kwargs.get('conv_bias', True),
            norm_op=arch_init_kwargs.get('norm_op', 'torch.nn.InstanceNorm3d'),
            norm_op_kwargs=arch_init_kwargs.get('norm_op_kwargs', {"eps": 1e-05, "affine": True}),
            dropout_op=arch_init_kwargs.get('dropout_op', None),
            dropout_op_kwargs=arch_init_kwargs.get('dropout_op_kwargs', None),
            nonlin=arch_init_kwargs.get('nonlin', 'torch.nn.LeakyReLU'),
            nonlin_kwargs=arch_init_kwargs.get('nonlin_kwargs', {"inplace": True}),
            deep_supervision=enable_deep_supervision
        )

        return network

    def _build_loss(self):
        """Override to use multi-task loss"""
        return MultiTaskLoss(
            seg_weight=self.seg_weight,
            cls_weight=self.cls_weight,
            loss_type=self.loss_type
        )

    def train_step(self, batch: dict) -> dict:
        """
        Custom training step for multi-task learning.
        """
        data = batch['data']
        target_seg = batch['target']  # Segmentation targets
        target_cls = batch.get('classification_target', None)  # Classification targets

        self.optimizer.zero_grad()

        # Forward pass
        output = self.network(data)

        # Multi-task output: segmentation and classification
        if isinstance(output, tuple) and len(output) == 2:
            seg_output, cls_output = output
        else:
            # Fallback if network returns only segmentation
            seg_output = output
            cls_output = None

        # Calculate loss
        loss_dict = self.loss(seg_output, target_seg, cls_output, target_cls)

        # Backward pass
        loss_dict['total_loss'].backward()
        self.optimizer.step()

        return loss_dict

    def validation_step(self, batch: dict) -> dict:
        """
        Custom validation step for multi-task learning.
        """
        data = batch['data']
        target_seg = batch['target']
        target_cls = batch.get('classification_target', None)

        with torch.no_grad():
            output = self.network(data)

            # Multi-task output
            if isinstance(output, tuple) and len(output) == 2:
                seg_output, cls_output = output
            else:
                seg_output = output
                cls_output = None

            # Calculate validation loss
            loss_dict = self.loss(seg_output, target_seg, cls_output, target_cls)

        return loss_dict

    def on_train_epoch_start(self):
        """
        Hook called at the start of each training epoch.
        Can be used for custom logic like dynamic loss weighting.
        """
        super().on_train_epoch_start()

        # Example: Dynamic loss weighting based on epoch
        if hasattr(self, 'current_epoch'):
            # Gradually increase classification weight
            epoch_ratio = min(self.current_epoch / 100.0, 1.0)  # Reach max at epoch 100
            self.loss.cls_weight = self.cls_weight * epoch_ratio

    def run_training(self):
        """
        Override the main training loop if needed for multi-task specific logic.
        """
        # You can add custom training logic here
        # For now, use the parent implementation
        super().run_training()