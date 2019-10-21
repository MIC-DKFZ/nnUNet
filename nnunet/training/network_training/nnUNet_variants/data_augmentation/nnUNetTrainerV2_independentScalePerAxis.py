from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_independentScalePerAxis(nnUNetTrainerV2):
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["independent_scale_factor_for_each_axis"] = True
