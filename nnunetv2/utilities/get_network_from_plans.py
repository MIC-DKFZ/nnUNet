from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op

from nnunetv2.utilities.label_handling.label_handling import get_labelmanager
from nnunetv2.utilities.network_initialization import InitWeights_He
from torch import nn


def get_network_from_plans(plans: dict, dataset_json: dict, configuration: str, num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    max_features = plans["configurations"][configuration]["unet_max_num_features"]
    initial_features = plans["configurations"][configuration]["UNet_base_num_features"]
    num_stages = len(plans["configurations"][configuration]["conv_kernel_sizes"])

    dim = len(plans["configurations"][configuration]["conv_kernel_sizes"][0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = get_labelmanager(plans, dataset_json)

    segmentation_network_class_name = plans["configurations"][configuration]["UNet_class_name"]
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
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
        'n_conv_per_stage' if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': plans["configurations"][configuration]['n_conv_per_stage_encoder'],
        'n_conv_per_stage_decoder': plans["configurations"][configuration]['n_conv_per_stage_decoder']
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
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    return model
