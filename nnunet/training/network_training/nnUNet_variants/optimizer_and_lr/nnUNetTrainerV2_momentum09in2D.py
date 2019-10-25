import torch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_momentum09in2D(nnUNetTrainerV2):
    def initialize_optimizer_and_scheduler(self):
        if self.threeD:
            momentum = 0.99
        else:
            momentum = 0.9
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=momentum, nesterov=True)
        self.lr_scheduler = None
