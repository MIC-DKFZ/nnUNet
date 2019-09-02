import torch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_Adam(nnUNetTrainerV2):

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        self.lr_scheduler = None
