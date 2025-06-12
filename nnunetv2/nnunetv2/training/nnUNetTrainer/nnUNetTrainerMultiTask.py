from cProfile import label
import torch
from torch import nn
import os
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.architectures.MultiTaskResEncUNet import MultiTaskResEncUNet, MultiTaskChannelAttentionResEncUNet, MultiTaskEfficientAttentionResEncUNet
from nnunetv2.training.loss.multitask_losses import MultiTaskLoss
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


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


    def get_tr_and_val_datasets(self):
        """Override to use dataset class with classification labels"""
        tr_keys, val_keys = self.do_split()
        dataset_name = self.plans_manager.dataset_name

        # Infer the appropriate dataset class (numpy or blosc2)
        dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # Use dataset class with classification labels enabled
        dataset_tr = dataset_class(
            self.preprocessed_dataset_folder, tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            load_subtype_labels=True,
            label_path=os.path.join(os.environ['nnUNet_raw'], dataset_name, "labels.csv")
        )
        dataset_val = dataset_class(
            self.preprocessed_dataset_folder, val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            load_subtype_labels=True,
            label_path=os.path.join(os.environ['nnUNet_raw'], dataset_name, "labels.csv")
        )

        return dataset_tr, dataset_val

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

        # Handle the import requirements for architecture kwargs
        import pydoc
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        # Map architecture class names to our custom classes
        architecture_mapping = {
            'nnunetv2.architectures.MultiTaskResEncUNet.MultiTaskResEncUNet': MultiTaskResEncUNet,
            'nnunetv2.architectures.MultiTaskResEncUNet.MultiTaskChannelAttentionResEncUNet': MultiTaskChannelAttentionResEncUNet,
            'nnunetv2.architectures.MultiTaskResEncUNet.MultiTaskEfficientAttentionResEncUNet': MultiTaskEfficientAttentionResEncUNet,
            # Add fallback for just the class name
            'MultiTaskResEncUNet': MultiTaskResEncUNet,
            'MultiTaskChannelAttentionResEncUNet': MultiTaskChannelAttentionResEncUNet,
            'MultiTaskEfficientAttentionResEncUNet': MultiTaskEfficientAttentionResEncUNet,
        }

        # Get the network class
        if architecture_class_name in architecture_mapping:
            network_class = architecture_mapping[architecture_class_name]
        else:
            # Fallback to default nnUNet behavior
            raise ValueError(f"Unknown architecture_class_name: {architecture_class_name}")

        # Create the network - note the different parameter names for multi-task networks
        network = network_class(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            # num_classification_classes=3,  # Update based on your classification classes
            **architecture_kwargs
        )

        # Initialize the network if it has an initialize method
        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

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
        import pdb
        pdb.set_trace()  # Debugging breakpoint
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)  # Segmentation targets
        target_cls = batch['classification_target'].to(self.device)  # Classification targets

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
        data = batch['data'].to(self.device)
        target_seg = batch['target'].to(self.device)  # Segmentation targets
        target_cls = batch['classification_target'].to(self.device)  # Classification targets

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
