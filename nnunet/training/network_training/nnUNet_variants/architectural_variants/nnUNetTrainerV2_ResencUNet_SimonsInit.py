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

from torch import nn

from nnunet.network_architecture.custom_modules.conv_blocks import BasicResidualBlock
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import \
    nnUNetTrainerV2_ResencUNet


def init_last_bn_before_add_to_0(module):
    if isinstance(module, BasicResidualBlock):
        module.norm2.weight = nn.init.constant_(module.norm2.weight, 0)
        module.norm2.bias = nn.init.constant_(module.norm2.bias, 0)


class nnUNetTrainerV2_ResencUNet_SimonsInit(nnUNetTrainerV2_ResencUNet):
    """
    SimonsInit = Simon Kohl's suggestion of initializing each residual block such that it adds nothing
    (weight and bias initialized to zero in last batch norm)
    """
    def initialize_network(self):
        ret = super().initialize_network()
        self.network.apply(init_last_bn_before_add_to_0)
        return ret


