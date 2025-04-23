from nnunetv2.training.nnUNetTrainer.variants.sparse_labels.nnUNetTrainer_betterIgnoreSampling import (
    nnUNetTrainer_betterIgnoreSampling_noSmooth,
)


class nnActiveTrainer_NoMirroring_2epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 2

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnActiveTrainer_NoMirroring_200epochs(nnActiveTrainer_NoMirroring_2epochs):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 200


class nnActiveTrainer_NoMirroring_500epochs(nnActiveTrainer_NoMirroring_2epochs):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 500


class nnActiveTrainer_NoMirroring_1000epochs(nnActiveTrainer_NoMirroring_2epochs):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 1000
