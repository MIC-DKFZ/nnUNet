import torch
from torch import nn
import numpy as np
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from src.architectures.MultiTaskResEncUNet import MultiTaskResEncUNet
from src.losses.multitask_losses import MultiTaskLoss
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainerMultiTask(nnUNetTrainerNoDeepSupervision):
    """Multi-task trainer for segmentation + classification"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Multi-task specific parameters
        self.num_classification_classes = 3  # Update based on your subtypes
        self.seg_weight = 1.0
        self.cls_weight = 0.5
        self.loss_type = 'dice_ce'  # Options: 'dice_ce', 'focal', 'tversky'

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                 num_input_channels, enable_deep_supervision: bool = False) -> nn.Module:

        num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = configuration_manager.UNet_class_name
        network_class = recursive_find_network_class(segmentation_network_class_name)

        if network_class is None:
            raise RuntimeError("Could not find network class")

        conv_or_blocks_per_stage = {
            'n_conv_per_stage' if network_class != ResidualEncoderUNet else 'n_blocks_per_stage':
            configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }

        # Use our custom multi-task network
        network = MultiTaskResEncUNet(
            input_channels=num_input_channels,
            num_segmentation_classes=label_manager.num_segmentation_classes,
            num_classification_classes=3,  # Your classification classes
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                  configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            **conv_or_blocks_per_stage,
            **configuration_manager.network_arch_class_kwargs
        )

        return network

    def _build_loss(self):
        """Override to use multi-task loss"""
        return MultiTaskLoss(
            seg_weight=self.seg_weight,
            cls_weight=self.cls_weight,
            loss_type=self.loss_type
        )

    def train_step(self, batch):
        """Override training step for multi-task learning"""
        data = batch['data']
        seg_target = batch['target']

        # Extract classification targets from filename or metadata
        # You'll need to implement this based on your data structure
        cls_target = self._extract_classification_targets(batch)

        data = data.to(self.device, non_blocking=True)
        seg_target = seg_target.to(self.device, non_blocking=True)
        cls_target = cls_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Forward pass
        with torch.cuda.amp.autocast(self.device.type == 'cuda') if self.device.type == 'cuda' else nullcontext():
            outputs = self.network(data)

            targets = {
                'segmentation': seg_target,
                'classification': cls_target
            }

            loss_dict = self.loss(outputs, targets)
            loss = loss_dict['total_loss']

        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return loss_dict

    def _extract_classification_targets(self, batch):
        """Extract classification labels from batch metadata"""
        # Implementation depends on how you store classification labels
        # Option 1: From filename
        batch_size = batch['data'].shape[0]
        cls_targets = []

        for i in range(batch_size):
            # Extract from keys or properties - adjust based on your data structure
            filename = batch['keys'][i]  # or however you access filenames

            if 'subtype0' in filename:
                cls_targets.append(0)
            elif 'subtype1' in filename:
                cls_targets.append(1)
            elif 'subtype2' in filename:
                cls_targets.append(2)
            else:
                cls_targets.append(0)  # default

        return torch.tensor(cls_targets, dtype=torch.long)

    def validation_step(self, batch):
        """Override validation step"""
        self.network.eval()

        data = batch['data']
        seg_target = batch['target']
        cls_target = self._extract_classification_targets(batch)

        data = data.to(self.device, non_blocking=True)
        seg_target = seg_target.to(self.device, non_blocking=True)
        cls_target = cls_target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(self.device.type == 'cuda') if self.device.type == 'cuda' else nullcontext():
                outputs = self.network(data)

                targets = {
                    'segmentation': seg_target,
                    'classification': cls_target
                }

                loss_dict = self.loss(outputs, targets)

        return loss_dict