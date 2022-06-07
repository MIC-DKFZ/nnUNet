from typing import Union

from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunet.network_architecture.generic_UNet import Generic_UNet
from torch import nn

from nnunetv2.training.nnunet_modules.nnUNetModule import nnUNetModule
from nnunetv2.utilities.network_initialization import InitWeights_He


class nnUNetModule_GenericUNet(nnUNetModule):
    def __init__(self, dataset_name_or_id: Union[int, str], plans_name: str, configuration: str, fold: int,
                 unpack_dataset: bool = True, folder_with_segs_from_previous_stage: str = None):
        super().__init__(dataset_name_or_id, plans_name, configuration, fold, unpack_dataset,
                         folder_with_segs_from_previous_stage)
        plans = self.plans
        initial_features = plans["configurations"][configuration]["UNet_base_num_features"]

        dim = len(plans["configurations"][configuration]["conv_kernel_sizes"][0])
        conv_op = convert_dim_to_conv_op(dim)

        strides = plans["configurations"][configuration]["pool_op_kernel_sizes"][1:]

        self.network = Generic_UNet(len(self.dataset_json["modality"]),
                                    initial_features,
                                    len(self.dataset_json["labels"]),
                                    len(strides),
                                    2,
                                    2,
                                    conv_op, get_matching_instancenorm(conv_op), {'eps': 1e-5, 'affine': True}, None,
                                                           None,
                                    nn.LeakyReLU, {'inplace': True}, True, False, lambda x: x, InitWeights_He(1e-2),
                                    strides, plans["configurations"][configuration]["conv_kernel_sizes"], False, True, True)
