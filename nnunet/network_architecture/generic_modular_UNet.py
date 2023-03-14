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
from nnunet.network_architecture.custom_modules.conv_blocks import StackedConvLayers
from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
import numpy as np
from torch.optim import SGD

from nnunet.backends import backend

"""
The idea behind this modular U-net ist that we decouple encoder and decoder and thus make things a) a lot more easy to 
combine and b) enable easy swapping between segmentation or classification mode of the same architecture
"""


def get_default_network_config(dim=2, dropout_p=None, nonlin="LeakyReLU", norm_type="bn"):
    """
    returns a dictionary that contains pointers to conv, nonlin and norm ops and the default kwargs I like to use
    :return:
    """
    props = {}
    if dim == 2:
        props['conv_op'] = nn.Conv2d
        props['dropout_op'] = nn.Dropout2d
    elif dim == 3:
        props['conv_op'] = nn.Conv3d
        props['dropout_op'] = nn.Dropout3d
    else:
        raise NotImplementedError

    if norm_type == "bn":
        if dim == 2:
            props['norm_op'] = nn.BatchNorm2d
        elif dim == 3:
            props['norm_op'] = nn.BatchNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    elif norm_type == "in":
        if dim == 2:
            props['norm_op'] = nn.InstanceNorm2d
        elif dim == 3:
            props['norm_op'] = nn.InstanceNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    else:
        raise NotImplementedError

    if dropout_p is None:
        props['dropout_op'] = None
        props['dropout_op_kwargs'] = {'p': 0, 'inplace': True}
    else:
        props['dropout_op_kwargs'] = {'p': dropout_p, 'inplace': True}

    props['conv_op_kwargs'] = {'stride': 1, 'dilation': 1, 'bias': False}  # kernel size will be set by network!

    if nonlin == "LeakyReLU":
        props['nonlin'] = nn.LeakyReLU
        props['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        props['nonlin'] = nn.ReLU
        props['nonlin_kwargs'] = {'inplace': True}
    else:
        raise ValueError

    return props


class PlainConvUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=480):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes (TODO)

        this one includes the bottleneck layer!

        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(PlainConvUNetEncoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)

        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this

        current_input_features = input_channels
        for stage in range(num_stages):
            current_output_features = min(int(base_num_features * feat_map_mul_on_downscale ** stage), max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            current_stage = StackedConvLayers(current_input_features, current_output_features, current_kernel_size,
                                              props, num_blocks_per_stage[stage], current_pool_kernel_size)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)
        self.output_features = current_output_features

    def forward(self, x, return_skips=None):
        """

        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        for s in self.stages:
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        if return_skips:
            return skips
        else:
            return x

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, pool_op_kernel_sizes, num_blocks_per_stage_encoder,
                                        feat_map_mul_on_downscale, batch_size):
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)

        tmp = num_blocks_per_stage_encoder[0] * np.prod(current_shape) * base_num_features \
              + num_modalities * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool + 1):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_blocks_per_stage_encoder[p]
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat
        return tmp * batch_size


class PlainConvUNetDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, network_props=None, deep_supervision=False,
                 upscale_logits=False):
        super(PlainConvUNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        """
        We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        """
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size

        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props

        if self.props['conv_op'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props['conv_op'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1

        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = len(previous_stages) - 1  # we have one less as the first stage here is what comes after the
        # bottleneck

        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []

        # only used for upsample_logits
        cum_upsample = np.cumprod(np.vstack(self.stage_pool_kernel_size), axis=0).astype(int)

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]

            self.tus.append(transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                                       previous_stage_pool_kernel_size[s + 1], bias=False))
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(StackedConvLayers(2 * features_skip, features_skip,
                                                 previous_stage_conv_op_kernel_size[s], self.props,
                                                 num_blocks_per_stage[i]))

            if deep_supervision and s != 0:
                seg_layer = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, bias=True)
                if upscale_logits:
                    upsample = Upsample(scale_factor=cum_upsample[s], mode=upsample_mode)
                    self.deep_supervision_outputs.append(nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(seg_layer)

        self.segmentation_output = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, bias=True)

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips, gt=None, loss=None):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips
        skips = skips[::-1]
        seg_outputs = []

        x = skips[0]  # this is the bottleneck

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)
            if self.deep_supervision and (i != len(self.tus) - 1):
                tmp = self.deep_supervision_outputs[i](x)
                if gt is not None:
                    tmp = loss(tmp, gt)
                seg_outputs.append(tmp)

        segmentation = self.segmentation_output(x)

        if self.deep_supervision:
            tmp = segmentation
            if gt is not None:
                tmp = loss(tmp, gt)
            seg_outputs.append(tmp)
            return seg_outputs[::-1]  # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            # the bottleneck of the UNet last
        else:
            return segmentation

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_classes, pool_op_kernel_sizes, num_blocks_per_stage_decoder,
                                        feat_map_mul_on_downscale, batch_size):
        """
        This only applies for num_blocks_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :return:
        """
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)
        tmp = (num_blocks_per_stage_decoder[-1] + 1) * np.prod(current_shape) * base_num_features + num_classes * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_blocks_per_stage_decoder[-(p+1)] + 1
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat

        return tmp * batch_size


class PlainConvUNet(SegmentationNetwork):
    use_this_for_batch_size_computation_2D = 1167982592.0
    use_this_for_batch_size_computation_3D = 1152286720.0

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None):
        super(PlainConvUNet, self).__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = PlainConvUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                            feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                            props, default_return_skips=True, max_num_features=max_features)
        self.decoder = PlainConvUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                            deep_supervision, upscale_logits)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, num_blocks_per_stage_encoder,
                                        num_blocks_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = PlainConvUNetEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                   num_modalities, pool_op_kernel_sizes,
                                                                   num_blocks_per_stage_encoder,
                                                                   feat_map_mul_on_downscale, batch_size)
        dec = PlainConvUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                   num_classes, pool_op_kernel_sizes,
                                                                   num_blocks_per_stage_decoder,
                                                                   feat_map_mul_on_downscale, batch_size)

        return enc + dec

    @staticmethod
    def compute_reference_for_vram_consumption_3d():
        patch_size = (160, 128, 128)
        pool_op_kernel_sizes = ((1, 1, 1),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2))
        conv_per_stage_encoder = (2, 2, 2, 2, 2, 2)
        conv_per_stage_decoder = (2, 2, 2, 2, 2)

        return PlainConvUNet.compute_approx_vram_consumption(patch_size, 32, 512, 4, 3, pool_op_kernel_sizes,
                                                             conv_per_stage_encoder, conv_per_stage_decoder, 2, 2)

    @staticmethod
    def compute_reference_for_vram_consumption_2d():
        patch_size = (256, 256)
        pool_op_kernel_sizes = (
            (1, 1), # (256, 256)
            (2, 2), # (128, 128)
            (2, 2), # (64, 64)
            (2, 2), # (32, 32)
            (2, 2), # (16, 16)
            (2, 2), # (8, 8)
            (2, 2)  # (4, 4)
        )
        conv_per_stage_encoder = (2, 2, 2, 2, 2, 2, 2)
        conv_per_stage_decoder = (2, 2, 2, 2, 2, 2)

        return PlainConvUNet.compute_approx_vram_consumption(patch_size, 32, 512, 4, 3, pool_op_kernel_sizes,
                                                             conv_per_stage_encoder, conv_per_stage_decoder, 2, 56)


if __name__ == "__main__":
    conv_op_kernel_sizes = ((3, 3),
                            (3, 3),
                            (3, 3),
                            (3, 3),
                            (3, 3),
                            (3, 3),
                            (3, 3))
    pool_op_kernel_sizes = ((1, 1),
                            (2, 2),
                            (2, 2),
                            (2, 2),
                            (2, 2),
                            (2, 2),
                            (2, 2))
    patch_size = (256, 256)
    batch_size = 56
    unet = backend.to(PlainConvUNet(4, 32, (2, 2, 2, 2, 2, 2, 2), 2, pool_op_kernel_sizes, conv_op_kernel_sizes,
                         get_default_network_config(2, dropout_p=None), 4, (2, 2, 2, 2, 2, 2), False, False, max_features=512))
    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)

    unet.compute_reference_for_vram_consumption_3d()
    unet.compute_reference_for_vram_consumption_2d()

    dummy_input = backend.to(torch.rand((batch_size, 4, *patch_size)))
    dummy_gt = backend.to((torch.rand((batch_size, 1, *patch_size)) * 4).round().clamp_(0, 3)).long()

    optimizer.zero_grad()
    skips = unet.encoder(dummy_input)
    print([i.shape for i in skips])
    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, {})
    output = unet.decoder(skips)

    l = loss(output, dummy_gt)
    l.backward()

    optimizer.step()

    import hiddenlayer as hl
    g = hl.build_graph(unet, dummy_input)
    g.save("/home/fabian/test.pdf")

    """conv_op_kernel_sizes = ((3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3))
    pool_op_kernel_sizes = ((1, 1, 1),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2))
    patch_size = (160, 128, 128)
    unet = PlainConvUNet(4, 32, (2, 2, 2, 2, 2, 2), 2, pool_op_kernel_sizes, conv_op_kernel_sizes,
                         get_default_network_config(3, dropout_p=None), 4, (2, 2, 2, 2, 2), False, False, max_features=512).cuda()
    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)

    unet.compute_reference_for_vram_consumption_3d()
    unet.compute_reference_for_vram_consumption_2d()

    dummy_input = torch.rand((2, 4, *patch_size)).cuda()
    dummy_gt = (torch.rand((2, 1, *patch_size)) * 4).round().clamp_(0, 3).cuda().long()

    optimizer.zero_grad()
    skips = unet.encoder(dummy_input)
    print([i.shape for i in skips])
    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, {})
    output = unet.decoder(skips)

    l = loss(output, dummy_gt)
    l.backward()

    optimizer.step()

    import hiddenlayer as hl
    g = hl.build_graph(unet, dummy_input)
    g.save("/home/fabian/test.pdf")"""
