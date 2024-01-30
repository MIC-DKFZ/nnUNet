from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.residual_unets.new_nnunet_presets.nnUNetPlannerXL import \
    nnUNetPlannerXL


class nnUNetPlannerXLx8(nnUNetPlannerXL):
    """
    Target is 8*40 GB VRAM max -> 8xA100 40GB or 4*A100 80GB
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetXLx8Plans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def plan_experiment(self):
        super(nnUNetPlannerXLx8, self).plan_experiment()
        for configuration in ['2d', '3d_fullres', '3d_lowres']:
            if configuration in self.plans['configurations']:
                self.plans['configurations'][configuration]['batch_size'] *= 8
        self.save_plans(self.plans)
        return self.plans
