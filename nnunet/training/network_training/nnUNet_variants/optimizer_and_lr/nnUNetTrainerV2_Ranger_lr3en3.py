from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.optimizer.ranger import Ranger


class nnUNetTrainerV2_Ranger_lr3en3(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 3e-3

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = Ranger(self.network.parameters(), self.initial_lr, k=6, N_sma_threshhold=5,
                                weight_decay=self.weight_decay)
        self.lr_scheduler = None

