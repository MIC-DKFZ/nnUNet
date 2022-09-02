from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.network_initialization import InitWeights_He


class nnUNetTrainerResEncUNet(nnUNetTrainer):
    # works on regular nnUNetPlans. Just for prototyping. We should plan with this architecture as well!
    def _get_network(self):
        max_features = self.plans["configurations"][self.configuration]["unet_max_num_features"]
        initial_features = self.plans["configurations"][self.configuration]["UNet_base_num_features"]
        num_stages = len(self.plans["configurations"][self.configuration]["conv_kernel_sizes"])

        dim = len(self.plans["configurations"][self.configuration]["conv_kernel_sizes"][0])
        conv_op = convert_dim_to_conv_op(dim)

        kwargs = {
                'conv_bias': True,
                'norm_op': get_matching_instancenorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True}
        }

        nnunet_resenc_unet_blocks_per_stage = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)

        # network class name!!
        model = ResidualEncoderUNet(
            input_channels=self.num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(initial_features * 2 ** i, max_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=self.plans["configurations"][self.configuration]["conv_kernel_sizes"],
            strides=self.plans["configurations"][self.configuration]["pool_op_kernel_sizes"],
            n_blocks_per_stage=nnunet_resenc_unet_blocks_per_stage[:num_stages],
            num_classes=self.label_manager.num_segmentation_heads,
            n_conv_per_stage_decoder=1,
            deep_supervision=True,
            **kwargs
        )
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)
        return model.to(self.device)
