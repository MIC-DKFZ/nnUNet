from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.residual_unet import ResidualEncoderUNet

from nnunetv2.experiment_planning.experiment_planners.residual_unets.ResEncUNet_planner import ResEncUNetPlanner


class nnUNetPlannerM(ResEncUNetPlanner):
    """
    Target is ~9-11 GB VRAM max -> older Titan, RTX 2080ti
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetMPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        gpu_memory_target_in_gb = 8
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 8

        # this is supposed to give the same GPU memory requirement as the default nnU-Net
        self.UNet_reference_val_3d = 600000000
        self.UNet_reference_val_2d = 133000000

