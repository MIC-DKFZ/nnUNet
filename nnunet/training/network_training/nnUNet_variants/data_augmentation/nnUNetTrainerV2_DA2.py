from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_DA2(nnUNetTrainerV2):
    def setup_DA_params(self):
        super().setup_DA_params()

        self.data_aug_params["independent_scale_factor_for_each_axis"] = True

        if self.threeD:
            self.data_aug_params["rotation_p_per_axis"] = 0.5
        else:
            self.data_aug_params["rotation_p_per_axis"] = 1

        self.data_aug_params["do_additive_brightness"] = True

