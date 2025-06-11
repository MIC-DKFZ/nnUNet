"""
Multi-task planner for nnUNet that extends ResEncUNet planner.

This planner generates plans for multi-task architectures that perform
both segmentation and classification tasks.
"""

import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.resencUNet_planner import ResEncUNetPlanner
from src.architectures.MultiTaskResEncUNet import MultiTaskResEncUNet


class MultiTaskResEncUNetPlanner(ResEncUNetPlanner):
    """
    Multi-task planner that extends ResEncUNet planner to support
    multi-task architectures for segmentation + classification.
    """

    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor',
                 plans_name: str = 'nnUNetMultiTaskResEncUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

        # Use our custom multi-task network
        self.UNet_class = MultiTaskResEncUNet

        # Adjust memory estimates for multi-task architecture
        # Multi-task networks need slightly more memory due to classification head
        self.UNet_reference_val_3d = int(680000000 * 1.1)  # 10% increase
        self.UNet_reference_val_2d = int(135000000 * 1.1)  # 10% increase

        # Overwrite default plans identifier to reflect multi-task
        self.plans_identifier = 'nnUNetMultiTaskResEncUNetPlans'

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        """
        Generate plans for multi-task configuration.

        This method extends the base ResEncUNet planner to use our custom
        multi-task architecture while maintaining all other functionality.
        """
        # Get the base plan from ResEncUNet planner
        plan = super().get_plans_for_configuration(
            spacing, median_shape, data_identifier,
            approximate_n_voxels_dataset, _cache
        )

        # Update the architecture to use our multi-task network
        plan['architecture']['network_class_name'] = (
            'src.architectures.MultiTaskResEncUNet.MultiTaskResEncUNet'
        )

        # Add multi-task specific parameters to architecture kwargs
        arch_kwargs = plan['architecture']['arch_kwargs']

        # Add classification parameters
        arch_kwargs['num_classification_classes'] = 3  # Adjust based on your dataset

        # Add any other multi-task specific parameters
        arch_kwargs['classification_dropout'] = 0.5
        arch_kwargs['use_classification_head'] = True

        return plan

    def static_estimate_VRAM_usage(self, patch_size, num_input_channels, num_output_channels,
                                   network_class_name, arch_kwargs, arch_kwargs_req_import):
        """
        Estimate VRAM usage for multi-task network.

        This method adjusts the VRAM estimation to account for the additional
        classification head in the multi-task architecture.
        """
        # Get base estimation
        base_estimate = super().static_estimate_VRAM_usage(
            patch_size, num_input_channels, num_output_channels,
            network_class_name, arch_kwargs, arch_kwargs_req_import
        )

        # Add overhead for classification head
        # Classification head is relatively small compared to segmentation network
        classification_overhead = 0.05  # 5% additional memory

        return base_estimate * (1 + classification_overhead)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        Generate data identifier for multi-task configurations.

        We use the same data as ResEncUNet since preprocessing is identical.
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # Reuse ResEncUNet data since preprocessing is the same
            return 'nnUNetResEncUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name

    def plan_experiment(self):
        """
        Plan the experiment for multi-task learning.

        This method extends the base planning to ensure compatibility
        with multi-task architectures.
        """
        # Call parent method to do the main planning
        super().plan_experiment()

        # Add multi-task specific metadata to plans
        if hasattr(self, 'plans'):
            self.plans['multitask_info'] = {
                'is_multitask': True,
                'tasks': ['segmentation', 'classification'],
                'num_classification_classes': 3,  # Adjust based on your dataset
                'architecture_type': 'MultiTaskResEncUNet'
            }
