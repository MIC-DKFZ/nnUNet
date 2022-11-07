from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerNoMirroring(nnUNetTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainer_onlyMirror01(nnUNetTrainer):
    """
    Only mirrors along spatial axes 0 and 1 for 3D and 0 for 2D
    """
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        patch_size = self.plans['configurations'][self.configuration]["patch_size"]
        dim = len(patch_size)
        if dim == 2:
            mirror_axes = (0, )
        else:
            mirror_axes = (0, 1)
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

