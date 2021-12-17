from torch import nn

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed
from typing import Union
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


class nnUNetTrainer(object):
    def __init__(self, dataset_name_or_id: Union[int, str], plans_name: str, configuration: str):
        self.dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        self.preprocessed_dataset_folder = join(nnUNet_preprocessed, self.dataset_name)
        self.plans_file = join(self.preprocessed_dataset_folder, plans_name + 'json')
        self.plans = load_json(self.plans_file)

        self.configuration = configuration

    def initialize(self, unpack_data: bool = True):
        pass

    def initialize_network(self):
        max_features = self.plans["configurations"][self.configuration]["unet_max_num_features"]
        initial_features = self.plans["configurations"][self.configuration]["UNet_base_num_features"]
        num_stages = len(self.plans["configurations"][self.configuration]["conv_kernel_sizes"])

        segmentation_network_class_name = self.plans["configurations"][self.configuration]["UNet_class_name"]
        # maybe just check that we are processing appropriate plans? Or how should we deal with different network classes?


        dim = len(self.plans["configurations"][self.configuration]["conv_kernel_sizes"][0])
        conv_op = convert_dim_to_conv_op(dim)
        norm_op = get_matching_instancenorm(conv_op)

        # network class name!!
        model = PlainConvUNet(
            input_channels=len(self.plans["dataset_json"]["modality"]),
            n_stages=num_stages,
            features_per_stage=[min(initial_features * 2**i, max_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=self.plans["configurations"][self.configuration]["conv_kernel_sizes"],
            strides=self.plans["configurations"][self.configuration]["pool_op_kernel_sizes"],
            n_conv_per_stage=2,
            num_classes=len(self.plans["dataset_json"]["labels"]),
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None, dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True},
            deep_supervision=True
        )



