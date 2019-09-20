import torch

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_momentum098(nnUNetTrainerV2):
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.98, nesterov=True)
        self.lr_scheduler = None
