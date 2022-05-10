from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from torch import nn


def get_network_from_plans(plans: dict, dataset_json: dict, configuration: str, deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings
    """
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

    # network class name!!
    model = network_class(
        input_channels=len(dataset_json["modality"]),
        n_stages=num_stages,
        features_per_stage=[min(initial_features * 2 ** i, max_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=plans["configurations"][configuration]["conv_kernel_sizes"],
        strides=plans["configurations"][configuration]["pool_op_kernel_sizes"],
        n_conv_per_stage=2,
        num_classes=len(dataset_json["labels"]),
        n_conv_per_stage_decoder=2,
        deep_supervision=deep_supervision,
        **kwargs[segmentation_network_class_name]
    )
    return model
