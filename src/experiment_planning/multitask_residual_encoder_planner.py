import torch
from typing import Union, List, Tuple
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import nnUNetPlannerResEncM
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import contextlib
from io import StringIO


class MultiTasknnUNetPlannerResEncM(nnUNetPlannerResEncM):
    """
    Custom planner for multitask ResEncUNet with shared encoder and dual decoders
    (segmentation + classification)
    """

    def __init__(self, dataset_name_or_id: Union[int, str],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor',
                 plans_name: str = 'MultiTasknnUNetResEncMPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False,
                 pretrained_checkpoint: str = None):

        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name,
                        plans_name, overwrite_target_spacing, suppress_transpose)

        self.pretrained_checkpoint = pretrained_checkpoint

    def plan_experiment(self):
        # Suppress print statements from parent class
        with contextlib.redirect_stdout(StringIO()):
            super().plan_experiment()

        # Add multitask specific configurations
        for configuration_name in self.plans['configurations'].keys():
            config = self.plans['configurations'][configuration_name]

            # ADDED: Smaller model size for testsing (Comment out to restore)
            config['architecture']['arch_kwargs']['features_per_stage'] = [16, 64, 128]
            config['architecture']['arch_kwargs']['n_stages'] = 3
            config['architecture']['arch_kwargs']['n_blocks_per_stage'] = [1, 3, 4] # same number as n_stage
            config['architecture']['arch_kwargs']['n_conv_per_stage_decoder'] = [1, 1] # one less than blocks per stage
            config['architecture']['arch_kwargs']['kernel_sizes'] = [
                [1, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ]
            config['architecture']['arch_kwargs']['strides'] = [
                [1, 1, 1],
                [1, 2, 2],
                [2, 2, 2],
            ]

            # Update architecture to use custom multitask network
            config['architecture']['network_class_name'] = 'src.architectures.multitask_resenc_unet.MultiTaskResEncUNet'

            # Add classification head configuration
            config['architecture']['classification_head'] = {
                'head_type': 'mlp',  # Use MLP head by default
                'num_classes': 3,  # subtype 0, 1, 2
                'dropout_rate': 0.3,
                'latent_dim': 1024,  # Large latent representation for expressiveness
                'mlp_hidden_dims': [512, 256],  # MLP hidden dimensions
                'use_all_features': False,  # Use last encoder stage for MLP
                # Legacy spatial attention config (kept for backward compatibility)
                'hidden_dims': [256, 128]
            }

            # Add multitask training configuration
            config['multitask_config'] = {
                'seg_weight': 0.6,  # Base weights (used when normalization is off)
                'cls_weight': 0.4,
                'use_focal_loss': True,
                'focal_gamma': 2.0,
                'focal_alpha': 0.25,
                # New normalization settings
                'use_loss_normalization': True,
                'normalization_warmup_epochs': 10,
                'progressive_weighting': True
            }

            # Optimizer selection based on model size and GPU memory
            config['optimizer_config'] = self._determine_optimizer_config(config)

            # Modify the data_identifier, plans_name is already overwritten
            config['data_identifier'] = f"{self.plans['plans_name']}_{configuration_name}"

            print(f'U-Net configuration: {configuration_name}')
            print(self.plans['configurations'][configuration_name])
            print()

        # Add pretrained checkpoint if specified
        if self.pretrained_checkpoint:
            self.plans['pretrained_checkpoint'] = self.pretrained_checkpoint

        # Save updated plans
        super().save_plans(self.plans)
        return self.plans

    def _determine_optimizer_config(self, config: dict) -> dict:
        """
        Determine optimizer based on estimated model size vs GPU VRAM
        """
        # Estimate model parameters based on architecture
        features_per_stage = config['architecture']['arch_kwargs']['features_per_stage']
        n_stages = config['architecture']['arch_kwargs']['n_stages']
        patch_size = config['patch_size']

        # Rough estimation of model size in MB
        # ResEncM encoder + segmentation decoder + classification head
        encoder_params = sum(f * f * 3 for f in features_per_stage) * 1e-6 * 4  # 4 bytes per float32
        decoder_params = encoder_params * 0.7  # Decoder typically smaller
        cls_head_params = features_per_stage[-1] * 256 * 4 * 1e-6  # Classification head

        estimated_model_size_mb = encoder_params + decoder_params + cls_head_params

        # Get available GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_memory_gb = self.gpu_memory_target_in_gb

        # Convert to MB and apply threshold
        gpu_memory_mb = gpu_memory_gb * 1024
        memory_threshold = 0.3 * gpu_memory_mb

        if estimated_model_size_mb > memory_threshold:
            optimizer_config = {
                'optimizer': 'SGD',
                'initial_lr': 1e-2,
                'momentum': 0.99,
                'weight_decay': 3e-5,
                'lr_scheduler': 'PolynomialLR'
            }
        else:
            optimizer_config = {
                'optimizer': 'Adam',
                'initial_lr': 1e-3,
                'weight_decay': 1e-5,
                'lr_scheduler': 'PolynomialLR'
            }

        return optimizer_config

    def determine_reader_writer_from_dataset_json(self, dataset_json: dict,
                                                **reader_writer_kwargs):
        """Override to handle multitask dataset structure"""
        rw = super().determine_reader_writer_from_dataset_json(dataset_json, **reader_writer_kwargs)

        # Add classification labels handling
        if 'classification_labels' in dataset_json:
            self.classification_labels = dataset_json['classification_labels']
        else:
            # Default to 3 subtypes based on task description
            self.classification_labels = {
                'subtype_0': 0,
                'subtype_1': 1,
                'subtype_2': 2
            }

        return rw

    def determine_segmentation_head_input_channels(self, configuration_manager: ConfigurationManager):
        """Override to account for shared encoder in multitask setup"""
        return super().determine_segmentation_head_input_channels(configuration_manager)

    def _get_plans_fname(self):
        """Custom plans filename for multitask"""
        return f"MultiTasknnUNetResEncMPlans__{self.dataset_name}.json"
