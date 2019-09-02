from nnunet.training.network_training.nnUNet_variants.optimizer_and_lr.nnUNetTrainerV2_Adam import nnUNetTrainerV2_Adam


class nnUNetTrainerV2_Adam_nnUNetTrainerlr(nnUNetTrainerV2_Adam):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 3e-4
