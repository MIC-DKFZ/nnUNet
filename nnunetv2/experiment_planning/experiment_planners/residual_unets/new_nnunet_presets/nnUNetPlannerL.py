from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.residual_unet import ResidualEncoderUNet

from nnunetv2.experiment_planning.experiment_planners.residual_unets.ResEncUNet_planner import ResEncUNetPlanner


class nnUNetPlannerL(ResEncUNetPlanner):
    """
    Target is ~24 GB VRAM max -> RTX 4090, Titan RTX, Quadro 6000
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        gpu_memory_target_in_gb = 24
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 24

        self.UNet_reference_val_3d = 2100000000  # 1840000000
        self.UNet_reference_val_2d = 380000000  # 352666667
        self.max_dataset_covered = 1

