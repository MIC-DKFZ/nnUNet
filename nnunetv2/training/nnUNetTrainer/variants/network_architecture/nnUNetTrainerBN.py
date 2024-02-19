<<<<<<< HEAD
from dynamic_network_architectures.architectures.residual_unet import ResidualEncoderUNet
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
=======
from typing import Union, Tuple, List

from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
>>>>>>> feature/improved_network_arch_definition_in_plans
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBN(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if 'norm_op' not in arch_init_kwargs.keys():
            raise RuntimeError("'norm_op' not found in arch_init_kwargs. This does not look like an architecture "
                               "I can hack BN into. This trainer only works with default nnU-Net architectures.")

        from pydoc import locate
        conv_op = locate(arch_init_kwargs['conv_op'])
        bn_class = get_matching_batchnorm(conv_op)
        arch_init_kwargs['norm_op'] = bn_class.__module__ + '.' + bn_class.__name__
        arch_init_kwargs['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}

        return nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels,
                                                        num_output_channels, enable_deep_supervision)

