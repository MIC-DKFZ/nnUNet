from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_NoNormalization import \
    nnUNetTrainerV2_NoNormalization


class nnUNetTrainerV2_NoNormalization_lr1en3(nnUNetTrainerV2_NoNormalization):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-3
