import warnings
from typing import Union
import torch

# Add ugly import guards to not break nnunet in case wrong dynamic-network-architectures version.
try:
    from dynamic_network_architectures.architectures.primus import PrimusS, PrimusM, PrimusL, PrimusB
except ImportError:
    warnings.warn(
        "Unable to import Primus architectures. Make sure you have the correct dynamic_network_architectures package installed."
    )
    PrimusS = None
    PrimusM = None
    PrimusL = None
    PrimusB = None
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


def get_network_from_name(
    arch_class_name,
    input_channels,
    output_channels,
    input_patchsize: tuple[int, int, int] | None = None,
    allow_init=True,
    deep_supervision: Union[bool, None] = None,
):
    if arch_class_name == "PrimusS":
        network = PrimusS(
            input_channels=input_channels,
            output_channels=output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=input_patchsize,
        )
    elif arch_class_name == "PrimusM":
        network = PrimusM(
            input_channels=input_channels,
            output_channels=output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=input_patchsize,
        )
    elif arch_class_name == "PrimusL":
        network = PrimusL(
            input_channels=input_channels,
            output_channels=output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=input_patchsize,
        )
    elif arch_class_name == "PrimusB":
        network = PrimusB(
            input_channels=input_channels,
            output_channels=output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=input_patchsize,
        )
    elif arch_class_name == "ResEncL":
        n_stages = 6
        network = ResidualEncoderUNet(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=output_channels,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )
    else:
        raise ValueError("Unknown architecture class name: {}".format(arch_class_name))

    if hasattr(network, "initialize") and allow_init:
        network.apply(network.initialize)

    return network


if __name__ == "__main__":
    import torch

    model = get_network_from_name("ResEncL", 1, 4, allow_init=False, deep_supervision=False)
    data = torch.rand((8, 1, 256, 256))
    target = torch.rand(size=(8, 1, 256, 256))
    outputs = model(data)  # this should be a list of torch.Tensor
