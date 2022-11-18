from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager


class nnUNetTrainerBN(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans, dataset_json, configuration, num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        label_manager = get_labelmanager(plans, dataset_json)

        max_features = plans["configurations"][configuration]["unet_max_num_features"]
        initial_features = plans["configurations"][configuration]["UNet_base_num_features"]
        num_stages = len(plans["configurations"][configuration]["conv_kernel_sizes"])

        dim = len(plans["configurations"][configuration]["conv_kernel_sizes"][0])
        conv_op = convert_dim_to_conv_op(dim)

        segmentation_network_class_name = plans["configurations"][configuration]["UNet_class_name"]
        mapping = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet
        }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accomodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': plans["configurations"][configuration][
                'n_conv_per_stage_encoder']
            if 'n_conv_per_stage_encoder' in plans["configurations"][configuration].keys() else 2,
            'n_conv_per_stage_decoder': plans["configurations"][configuration]['n_conv_per_stage_decoder']
            if 'n_conv_per_stage_decoder' in plans["configurations"][configuration].keys() else 2
        }

        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(initial_features * 2 ** i, max_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=plans["configurations"][configuration]["conv_kernel_sizes"],
            strides=plans["configurations"][configuration]["pool_op_kernel_sizes"],
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=True,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        return model
