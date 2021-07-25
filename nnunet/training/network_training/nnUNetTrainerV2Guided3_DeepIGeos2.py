from nnunet.training.network_training.nnUNetTrainerV2Guided3_DeepIGeos1 import nnUNetTrainerV2Guided3_DeepIGeos1
from nnunet.training.loss_functions.dice_loss import ATM_and_DC_and_CE_loss

class nnUNetTrainerV2Guided3_DeepIGeos2(nnUNetTrainerV2Guided3_DeepIGeos1):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, deep_i_geos_value=0.0):
        self.deep_i_geos_value = deep_i_geos_value
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1500
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        # self.loss = ATM_and_DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        self.pin_memory = True