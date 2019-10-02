from nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_Loss_Dice import \
    nnUNetTrainerV2_Loss_Dice


class nnUNetTrainerV2_Loss_Dice_LR1en3(nnUNetTrainerV2_Loss_Dice):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-3
