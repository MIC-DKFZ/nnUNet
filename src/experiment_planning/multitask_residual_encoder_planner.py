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
                 pretrained_checkpoint: str = None,
                 head_type: str = 'latent_spatial'):

        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name,
                        plans_name, overwrite_target_spacing, suppress_transpose)

        self.pretrained_checkpoint = pretrained_checkpoint
        self.head_type = head_type

    def plan_experiment(self):
        # Suppress print statements from parent class
        with contextlib.redirect_stdout(StringIO()):
            super().plan_experiment()

        # Add multitask specific configurations
        for configuration_name in self.plans['configurations'].keys():
            config = self.plans['configurations'][configuration_name]

            # 4x scaled model size for enhanced capacity
            if configuration_name == '2d':
                config['architecture']['arch_kwargs']['features_per_stage'] = [32, 64, 128]
                config['architecture']['arch_kwargs']['n_stages'] = 3
                config['architecture']['arch_kwargs']['n_blocks_per_stage'] = [1, 1, 2]
                config['architecture']['arch_kwargs']['n_conv_per_stage_decoder'] = [1, 1]
                config['architecture']['arch_kwargs']['kernel_sizes'] = [
                    [3, 3],
                    [3, 3],
                    [3, 3],
                ]
                config['architecture']['arch_kwargs']['strides'] = [
                    [1, 1],
                    [2, 2],
                    [2, 2],
                ]
                # Reduce batch size for larger model
                config['batch_size'] = 12
            else:  # 3d_fullres
                config['architecture']['arch_kwargs']['features_per_stage'] = [16, 64]
                config['architecture']['arch_kwargs']['n_stages'] = 2
                config['architecture']['arch_kwargs']['n_blocks_per_stage'] = [1, 1]
                config['architecture']['arch_kwargs']['n_conv_per_stage_decoder'] = [1]
                config['architecture']['arch_kwargs']['kernel_sizes'] = [
                    [1, 3, 3],
                    [3, 3, 3],
                ]
                config['architecture']['arch_kwargs']['strides'] = [
                    [1, 1, 1],
                    [2, 2, 2],
                ]
                # Reduce batch size for larger 3D model
                config['batch_size'] = 2

            # Update architecture to use custom multitask network
            config['architecture']['network_class_name'] = 'src.architectures.multitask_resenc_unet.MultiTaskResEncUNet'

            # Add latent layer configuration (4x scaled)
            last_stage_channels = config['architecture']['arch_kwargs']['features_per_stage'][-1]
            config['architecture']['latent_layer'] = {
                'channels': last_stage_channels,
                'spatial_size': [24, 24] if configuration_name == '2d' else [16, 32, 32],
                'compression_ratio': 0.5,
                'compression_channels': last_stage_channels // 2,
                'activation': 'torch.nn.LeakyReLU',
                'use_bottleneck': True,
                'bottleneck_reduction': 2,
                'normalization': 'torch.nn.modules.instancenorm.InstanceNorm2d' if configuration_name == '2d' else 'torch.nn.modules.instancenorm.InstanceNorm3d',
                'dropout_rate': 0.1
            }

            # Add classification head configuration based on head type
            head_type = getattr(self, 'head_type', 'latent_spatial')  # Default to latent_spatial
            config['architecture']['head_type'] = head_type

            if head_type == 'simple_mlp':
                # Simple MLP head - no latent layer, direct from encoder
                config['architecture']['classification_head'] = {
                    'num_classes': 3,
                    'mlp_hidden_dims': [256, 128],  # Simple hidden dimensions
                    'dropout_rate': 0.3,
                    'global_pooling': 'adaptive_avg'
                }
                # Remove latent layer for simple_mlp
                config['architecture'].pop('latent_layer', None)
            else:
                # Standard latent_spatial head configuration
                if configuration_name == '2d':
                    config['architecture']['classification_head'] = {
                        'num_classes': 3,
                        'dropout_rate': 0.2,
                        'initial_conv_config': {
                            'input_channels': last_stage_channels,
                            'output_channels': last_stage_channels * 2,
                            'kernel_size': [3, 3],
                            'stride': [1, 1],
                            'padding': [1, 1],
                            'use_batch_norm': True,
                            'activation': 'torch.nn.LeakyReLU'
                        },
                        'conv_layers': [
                            {
                                'in_channels': last_stage_channels * 2,
                                'out_channels': last_stage_channels,
                                'kernel_size': [3, 3],
                                'stride': [2, 2],
                                'padding': [1, 1],
                                'use_batch_norm': True,
                                'activation': 'torch.nn.LeakyReLU',
                                'dropout_rate': 0.1
                            }
                        ],
                        'global_pooling': 'adaptive_avg',
                        'hidden_dims': [last_stage_channels * 2, last_stage_channels],
                        'use_all_features': False,
                        'feature_fusion': {
                            'enabled': False,
                            'fusion_type': 'concatenation',
                            'skip_connections': []
                        }
                    }
                else:  # 3d_fullres
                    config['architecture']['classification_head'] = {
                        'num_classes': 3,
                        'dropout_rate': 0.2,
                        'initial_conv_config': {
                            'input_channels': last_stage_channels,
                            'output_channels': last_stage_channels * 2,
                            'kernel_size': [3, 3, 3],
                            'stride': [1, 1, 1],
                            'padding': [1, 1, 1],
                            'use_batch_norm': True,
                            'activation': 'torch.nn.LeakyReLU'
                        },
                        'conv_layers': [
                            {
                                'in_channels': last_stage_channels * 2,
                                'out_channels': last_stage_channels,
                                'kernel_size': [3, 3, 3],
                                'stride': [2, 2, 2],
                                'padding': [1, 1, 1],
                                'use_batch_norm': True,
                                'activation': 'torch.nn.LeakyReLU',
                                'dropout_rate': 0.1
                            }
                        ],
                        'global_pooling': 'adaptive_avg',
                        'hidden_dims': [last_stage_channels * 2, last_stage_channels],
                        'use_all_features': False,
                        'feature_fusion': {
                            'enabled': False,
                            'fusion_type': 'concatenation',
                            'skip_connections': []
                        }
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
