from typing import Union, List, Tuple

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import \
    nnUNetPlannerResEncL
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet


class nnUNetPlannerResEncL_torchres(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlans_torchres',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_torch_fornnunet
        resampling_data_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        resampling_seg = resample_torch_fornnunet
        resampling_seg_kwargs = {
            "is_seg": True,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_torch_fornnunet
        resampling_fn_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        return resampling_fn, resampling_fn_kwargs


class nnUNetPlannerResEncL_torchres_sepz(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlans_torchres_sepz',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_torch_fornnunet
        resampling_data_kwargs = {
            "is_seg": False,
            'force_separate_z': None,
            'memefficient_seg_resampling': False,
            'separate_z_anisotropy_threshold': ANISO_THRESHOLD
        }
        resampling_seg = resample_torch_fornnunet
        resampling_seg_kwargs = {
            "is_seg": True,
            'force_separate_z': None,
            'memefficient_seg_resampling': False,
            'separate_z_anisotropy_threshold': ANISO_THRESHOLD
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_torch_fornnunet
        resampling_fn_kwargs = {
            "is_seg": False,
            'force_separate_z': None,
            'memefficient_seg_resampling': False,
            'separate_z_anisotropy_threshold': ANISO_THRESHOLD
        }
        return resampling_fn, resampling_fn_kwargs


class nnUNetPlanner_torchres(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlans_torchres',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_torch_fornnunet
        resampling_data_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        resampling_seg = resample_torch_fornnunet
        resampling_seg_kwargs = {
            "is_seg": True,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_torch_fornnunet
        resampling_fn_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        return resampling_fn, resampling_fn_kwargs
