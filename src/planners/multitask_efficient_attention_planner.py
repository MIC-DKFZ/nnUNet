"""
Multi-task planner with efficient attention for nnUNet.

This planner generates plans for multi-task architectures with efficient attention
that perform both segmentation and classification tasks.
"""

import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from src.planners.multitask_base_planner import MultiTaskResEncUNetPlanner
from src.architectures.MultiTaskResEncUNet import MultiTaskEfficientAttentionResEncUNet


class MultiTaskEfficientAttentionResEncUNetPlanner(MultiTaskResEncUNetPlanner):
    """
    Multi-task planner with efficient attention that extends the base multi-task planner
    to support efficient attention mechanisms in multi-task architectures.
    """

    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor',
                 plans_name: str = 'nnUNetMultiTaskEfficientAttentionResEncUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

        # Use our custom multi-task network with efficient attention
        self.UNet_class = MultiTaskEfficientAttentionResEncUNet

        # Adjust memory estimates for efficient attention
        self.UNet_reference_val_3d = int(680000000 * 1.12)  # 12% increase
        self.UNet_reference_val_2d = int(135000000 * 1.12)  # 12% increase

        self.plans_identifier = 'nnUNetMultiTaskEfficientAttentionResEncUNetPlans'

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        """
        Generate plans for multi-task configuration with efficient attention.
        """
        # Get the base plan from multi-task planner
        plan = super().get_plans_for_configuration(
            spacing, median_shape, data_identifier,
            approximate_n_voxels_dataset, _cache
        )

        # Update the architecture to use our efficient attention multi-task network
        plan['architecture']['network_class_name'] = (
            'src.architectures.MultiTaskResEncUNet.MultiTaskEfficientAttentionResEncUNet'
        )

        # Add efficient attention specific parameters
        arch_kwargs = plan['architecture']['arch_kwargs']
        arch_kwargs['use_efficient_attention'] = True
        arch_kwargs['attention_heads'] = 8
        arch_kwargs['attention_dim'] = 64

        return plan

    def static_estimate_VRAM_usage(self, patch_size, num_input_channels, num_output_channels,
                                   network_class_name, arch_kwargs, arch_kwargs_req_import):
        """
        Estimate VRAM usage for multi-task network with efficient attention.
        """
        # Get base estimation
        base_estimate = super().static_estimate_VRAM_usage(
            patch_size, num_input_channels, num_output_channels,
            network_class_name, arch_kwargs, arch_kwargs_req_import
        )

        # Efficient attention has lower overhead than standard attention
        attention_overhead = 0.02  # 2% additional memory

        return base_estimate * (1 + attention_overhead)

    def plan_experiment(self):
        """
        Plan the experiment for multi-task learning with efficient attention.
        """
        # Call parent method to do the main planning
        super().plan_experiment()

        # Add efficient attention specific metadata to plans
        if hasattr(self, 'plans'):
            self.plans['multitask_info']['architecture_type'] = 'MultiTaskEfficientAttentionResEncUNet'
            self.plans['multitask_info']['attention_mechanism'] = 'efficient_attention'
            self.plans['multitask_info']['attention_heads'] = 8
            self.plans['multitask_info']['attention_dim'] = 64
