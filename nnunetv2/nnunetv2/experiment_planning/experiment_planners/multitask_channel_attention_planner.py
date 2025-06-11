"""
Multi-task planner with channel attention for nnUNet.

This planner generates plans for multi-task architectures with channel attention
that perform both segmentation and classification tasks.
"""

import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.multitask_base_planner import MultiTaskResEncUNetPlanner
from nnunetv2.architectures.MultiTaskResEncUNet import MultiTaskChannelAttentionResEncUNet


class MultiTaskChannelAttentionResEncUNetPlanner(MultiTaskResEncUNetPlanner):
    """
    Multi-task planner with channel attention that extends the base multi-task planner
    to support channel attention mechanisms in multi-task architectures.
    """

    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor',
                 plans_name: str = 'nnUNetMultiTaskChannelAttentionResEncUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

        # Use our custom multi-task network with channel attention
        self.UNet_class = MultiTaskChannelAttentionResEncUNet

        # Channel attention adds some overhead, adjust memory estimates
        self.UNet_reference_val_3d = int(680000000 * 1.15)  # 15% increase
        self.UNet_reference_val_2d = int(135000000 * 1.15)  # 15% increase

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        """
        Generate plans for multi-task configuration with channel attention.

        This method extends the base multi-task planner to use our custom
        channel attention multi-task architecture.
        """
        # Get the base plan from multi-task planner
        plan = super().get_plans_for_configuration(
            spacing, median_shape, data_identifier,
            approximate_n_voxels_dataset, _cache
        )

        # Update the architecture to use our channel attention multi-task network
        plan['architecture']['network_class_name'] = (
            'src.architectures.MultiTaskResEncUNet.MultiTaskChannelAttentionResEncUNet'
        )

        # Add channel attention specific parameters
        arch_kwargs = plan['architecture']['arch_kwargs']
        arch_kwargs['use_channel_attention'] = True
        arch_kwargs['attention_reduction_ratio'] = 16  # Standard SE-Net ratio

        return plan

    def static_estimate_VRAM_usage(self, patch_size, num_input_channels, num_output_channels,
                                   network_class_name, arch_kwargs, arch_kwargs_req_import):
        """
        Estimate VRAM usage for multi-task network with channel attention.

        Channel attention adds additional parameters and computations.
        """
        # Get base estimation
        base_estimate = super().static_estimate_VRAM_usage(
            patch_size, num_input_channels, num_output_channels,
            network_class_name, arch_kwargs, arch_kwargs_req_import
        )

        # Add overhead for channel attention mechanisms
        # Channel attention adds relatively small overhead
        attention_overhead = 0.03  # 3% additional memory for attention

        return base_estimate * (1 + attention_overhead)

    def plan_experiment(self):
        """
        Plan the experiment for multi-task learning with channel attention.
        """
        # Call parent method to do the main planning
        super().plan_experiment()

        # Add channel attention specific metadata to plans
        if hasattr(self, 'plans'):
            self.plans['multitask_info']['architecture_type'] = 'MultiTaskChannelAttentionResEncUNet'
            self.plans['multitask_info']['attention_mechanism'] = 'channel_attention'
            self.plans['multitask_info']['attention_reduction_ratio'] = 16
