from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.residual_unets_moreFilt.nnUNetPlannerXLmoreFilt import \
    nnUNetPlannerXLmoreFilt


class nnUNetPlannerXLx8moreFilt(nnUNetPlannerXLmoreFilt):
    """
    Target is 8*40 GB VRAM max -> 8xA100 40GB or 4*A100 80GB
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,  # this needs to be 40 as we lan for the same size per GPU as XL
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetXLx8moreFiltPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        gpu_memory_target_in_gb = 40
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def plan_experiment(self):
        print('DO NOT TRUST ANY PRINTED PLANS AS THE BATCH SIZE WILL NOT YET HAVE BEEN INCREASED! FINAL BATCH SIZE IS '
              '8x OF WHAT YOU SEE')
        super(nnUNetPlannerXLmoreFilt, self).plan_experiment()
        for configuration in ['2d', '3d_fullres', '3d_lowres']:
            if configuration in self.plans['configurations']:
                self.plans['configurations'][configuration]['batch_size'] *= 8
        self.save_plans(self.plans)
        return self.plans
