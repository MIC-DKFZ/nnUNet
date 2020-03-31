#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_reduceMomentumDuringTraining(nnUNetTrainerV2):
    """
    This implementation will not work with LR scheduler!!!!!!!!!!

    After epoch 800, linearly decrease momentum from 0.99 to 0.9
    """
    def initialize_optimizer_and_scheduler(self):
        current_momentum = 0.99
        min_momentum = 0.9

        if self.epoch > 800:
            current_momentum = current_momentum - (current_momentum - min_momentum) / 200 * (self.epoch - 800)

        self.print_to_log_file("current momentum", current_momentum)
        assert self.network is not None, "self.initialize_network must be called first"
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)
        else:
            # can't reinstantiate because that would break NVIDIA AMP
            self.optimizer.param_groups[0]["momentum"] = current_momentum
        self.lr_scheduler = None

    def on_epoch_end(self):
        self.initialize_optimizer_and_scheduler()
        return super().on_epoch_end()
