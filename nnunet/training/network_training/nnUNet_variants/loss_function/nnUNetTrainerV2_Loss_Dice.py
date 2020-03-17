from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainerV2_Loss_Dice(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = SoftDiceLoss(**{'apply_nonlin': softmax_helper, 'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False})


class nnUNetTrainerV2_Loss_DicewithBG(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = SoftDiceLoss(**{'apply_nonlin': softmax_helper, 'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': True})

