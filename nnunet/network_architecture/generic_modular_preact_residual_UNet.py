import numpy as np
from copy import deepcopy
import torch

from nnunet.backends import backend

from torch.nn import Identity

from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.generic_modular_UNet import PlainConvUNetDecoder, get_default_network_config
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
from torch.optim import SGD


class BasicPreActResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is norm nonlin conv norm nonlin conv
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = stride
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        self.norm1 = props['norm_op'](in_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])
        self.conv1 = props['conv_op'](in_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **kwargs_conv1)

        if props['dropout_op_kwargs']['p'] != 0:
            self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.dropout = Identity()

        self.norm2 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])
        self.conv2 = props['conv_op'](out_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])

        if (self.stride is not None and any((i != 1 for i in self.stride))) or (in_planes != out_planes):
            stride_here = stride if stride is not None else 1
            self.downsample_skip = nn.Sequential(props['conv_op'](in_planes, out_planes, 1, stride_here, bias=False))
        else:
            self.downsample_skip = None

    def forward(self, x):
        residual = x

        out = self.nonlin1(self.norm1(x))

        if self.downsample_skip is not None:
            residual = self.downsample_skip(out)

        # norm nonlin conv
        out = self.conv1(out)

        out = self.dropout(out) # this does nothing if props['dropout_op_kwargs'] == 0

        # norm nonlin conv
        out = self.conv2(self.nonlin2(self.norm2(out)))

        out += residual

        return out


class PreActResidualLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None):
        super().__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            BasicPreActResidualBlock(input_channels, output_channels, kernel_size, network_props, first_stride),
            *[BasicPreActResidualBlock(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)


class PreActResidualUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=480, pool_type: str = 'conv'):
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
        super(PreActResidualUNetEncoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        pool_op = self._handle_pool(pool_type)

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

        self.initial_conv = props['conv_op'](input_channels, base_num_features, 3, padding=1, **props['conv_op_kwargs'])

        current_input_features = base_num_features
        for stage in range(num_stages):
            current_output_features = min(base_num_features * feat_map_mul_on_downscale ** stage, max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]

            current_pool_kernel_size = pool_op_kernel_sizes[stage]
            if pool_op is not None:
                pool_kernel_size_for_conv = [1 for i in current_pool_kernel_size]
            else:
                pool_kernel_size_for_conv = current_pool_kernel_size

            current_stage = PreActResidualLayer(current_input_features, current_output_features, current_kernel_size, props,
                                                self.num_blocks_per_stage[stage], pool_kernel_size_for_conv)
            if pool_op is not None:
                current_stage = nn.Sequential(pool_op(current_pool_kernel_size), current_stage)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)
        self.output_features = current_input_features

    def _handle_pool(self, pool_type):
        assert pool_type in ['conv', 'avg', 'max']
        if pool_type == 'avg':
            if self.props['conv_op'] == nn.Conv2d:
                pool_op = nn.AvgPool2d
            elif self.props['conv_op'] == nn.Conv3d:
                pool_op = nn.AvgPool3d
            else:
                raise NotImplementedError
        elif pool_type == 'max':
            if self.props['conv_op'] == nn.Conv2d:
                pool_op = nn.MaxPool2d
            elif self.props['conv_op'] == nn.Conv3d:
                pool_op = nn.MaxPool3d
            else:
                raise NotImplementedError
        elif pool_type == 'conv':
            pool_op = None
        else:
            raise ValueError
        return pool_op

    def forward(self, x, return_skips=None):
        """

        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        x = self.initial_conv(x)

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
                                        num_modalities, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        feat_map_mul_on_downscale, batch_size):
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)

        tmp = (num_conv_per_stage_encoder[0] * 2 + 1) * np.prod(current_shape) * base_num_features \
              + num_modalities * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool + 1):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_conv_per_stage_encoder[p] * 2 + 1 # + 1 for conv in skip in first block
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat
        return tmp * batch_size


class PreActResidualUNetDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, network_props=None, deep_supervision=False,
                 upscale_logits=False):
        super(PreActResidualUNetDecoder, self).__init__()
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
            self.stages.append(PreActResidualLayer(2 * features_skip, features_skip, previous_stage_conv_op_kernel_size[s],
                                                   self.props, num_blocks_per_stage[i], None))

            if deep_supervision and s != 0:
                norm = self.props['norm_op'](features_skip, **self.props['norm_op_kwargs'])
                nonlin = self.props['nonlin'](**self.props['nonlin_kwargs'])
                seg_layer = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, bias=True)
                if upscale_logits:
                    upsample = Upsample(scale_factor=cum_upsample[s], mode=upsample_mode)
                    self.deep_supervision_outputs.append(nn.Sequential(norm, nonlin, seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(nn.Sequential(norm, nonlin, seg_layer))

        self.segmentation_conv_norm = self.props['norm_op'](features_skip, **self.props['norm_op_kwargs'])
        self.segmentation_conv_nonlin = self.props['nonlin'](**self.props['nonlin_kwargs'])
        self.segmentation_output = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, bias=True)
        self.segmentation_output = nn.Sequential(self.segmentation_conv_norm, self.segmentation_conv_nonlin,
                                                 self.segmentation_output)

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips):
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
                seg_outputs.append(self.deep_supervision_outputs[i](x))

        segmentation = self.segmentation_output(x)

        if self.deep_supervision:
            seg_outputs.append(segmentation)
            return seg_outputs[::-1]  # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            # the bottleneck of the UNet last
        else:
            return segmentation

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_classes, pool_op_kernel_sizes, num_blocks_per_stage_decoder,
                                        feat_map_mul_on_downscale, batch_size):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
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
        tmp = (num_blocks_per_stage_decoder[-1] * 2 + 1) * np.prod(current_shape) * base_num_features + num_classes * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_blocks_per_stage_decoder[-(p + 1)] * 2 + 1 + 1 # +1 for transpconv and +1 for conv in skip
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat

        return tmp * batch_size


class PreActResidualUNet(SegmentationNetwork):
    use_this_for_batch_size_computation_2D = 858931200.0  # 1167982592.0
    use_this_for_batch_size_computation_3D = 727842816.0  # 1152286720.0

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None):
        super(PreActResidualUNet, self).__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = PreActResidualUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                                 feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                                 props, default_return_skips=True, max_num_features=max_features)
        self.decoder = PreActResidualUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                                 deep_supervision, upscale_logits)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        num_conv_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = PreActResidualUNetEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                        num_modalities, pool_op_kernel_sizes,
                                                                        num_conv_per_stage_encoder,
                                                                        feat_map_mul_on_downscale, batch_size)
        dec = PreActResidualUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                        num_classes, pool_op_kernel_sizes,
                                                                        num_conv_per_stage_decoder,
                                                                        feat_map_mul_on_downscale, batch_size)

        return enc + dec

    @staticmethod
    def compute_reference_for_vram_consumption_3d():
        patch_size = (128, 128, 128)
        pool_op_kernel_sizes = ((1, 1, 1),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2))
        blocks_per_stage_encoder = (1, 1, 1, 1, 1, 1)
        blocks_per_stage_decoder = (1, 1, 1, 1, 1)

        return PreActResidualUNet.compute_approx_vram_consumption(patch_size, 20, 512, 4, 3, pool_op_kernel_sizes,
                                                                  blocks_per_stage_encoder, blocks_per_stage_decoder, 2, 2)

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
        blocks_per_stage_encoder = (1, 1, 1, 1, 1, 1, 1)
        blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1)

        return PreActResidualUNet.compute_approx_vram_consumption(patch_size, 20, 512, 4, 3, pool_op_kernel_sizes,
                                                                  blocks_per_stage_encoder, blocks_per_stage_decoder, 2, 50)


class FabiansPreActUNet(SegmentationNetwork):
    use_this_for_2D_configuration = 1792460800
    use_this_for_3D_configuration = 1318592512
    default_blocks_per_stage_encoder = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
    default_blocks_per_stage_decoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    default_min_batch_size = 2 # this is what works with the numbers above

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None):
        super().__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = PreActResidualUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                           feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                           props, default_return_skips=True, max_num_features=max_features)
        props['dropout_op_kwargs']['p'] = 0
        self.decoder = PlainConvUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                           deep_supervision, upscale_logits)

        expected_num_skips = len(conv_kernel_sizes) - 1
        num_features_skips = [min(max_features, base_num_features * 2**i) for i in range(expected_num_skips)]
        norm_nonlins = []
        for i in range(expected_num_skips):
            norm_nonlins.append(nn.Sequential(props['norm_op'](num_features_skips[i], **props['norm_op_kwargs']), props['nonlin'](**props['nonlin_kwargs'])))
        self.norm_nonlins = nn.ModuleList(norm_nonlins)

        if initializer is not None:
            self.apply(initializer)

    def forward(self, x, gt=None, loss=None):
        skips = self.encoder(x)
        for i, op in enumerate(self.norm_nonlins):
            skips[i] = self.norm_nonlins[i](skips[i])
        return self.decoder(skips, gt, loss)

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, num_blocks_per_stage_encoder,
                                        num_blocks_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = PreActResidualUNetEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                        num_modalities, pool_op_kernel_sizes,
                                                                        num_blocks_per_stage_encoder,
                                                                        feat_map_mul_on_downscale, batch_size)
        dec = PlainConvUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                   num_classes, pool_op_kernel_sizes,
                                                                   num_blocks_per_stage_decoder,
                                                                   feat_map_mul_on_downscale, batch_size)

        return enc + dec


def find_3d_configuration():
    backend.set_benchmark(True)
    backend.set_deterministic(False)

    conv_op_kernel_sizes = ((3, 3, 3),
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

    patch_size = (128, 128, 128)
    base_num_features = 32
    input_modalities = 4
    blocks_per_stage_encoder = (1, 3, 4, 6, 6, 6)
    blocks_per_stage_decoder = (2, 2, 2, 2, 2)
    feat_map_mult_on_downscale = 2
    num_classes = 5
    max_features = 320
    batch_size = 2

    unet = backend.to(FabiansPreActUNet(input_modalities, base_num_features, blocks_per_stage_encoder, feat_map_mult_on_downscale,
    pool_op_kernel_sizes, conv_op_kernel_sizes, get_default_network_config(3, dropout_p=None), num_classes,
    blocks_per_stage_decoder, True, False, max_features=max_features))

    scaler = backend.get_gradscaler()
    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)

    print(unet.compute_approx_vram_consumption(patch_size, base_num_features, max_features, input_modalities,
                                               num_classes, pool_op_kernel_sizes, blocks_per_stage_encoder,
                                               blocks_per_stage_decoder, feat_map_mult_on_downscale, batch_size))

    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    dummy_input = backend.to(torch.rand((batch_size, input_modalities, *patch_size)))
    dummy_gt = backend.to((torch.rand((batch_size, 1, *patch_size)) * num_classes).round().clamp_(0, num_classes-1)).long()

    for i in range(10):
        optimizer.zero_grad()

        with backend.autocast():
            skips = unet.encoder(dummy_input)
            print([i.shape for i in skips])
            output = unet.decoder(skips)[0]

            l = loss(output, dummy_gt)
            print(l.item())
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

    with backend.autocast():
        import hiddenlayer as hl
        g = hl.build_graph(unet, dummy_input, transforms=None)
        g.save("/home/fabian/test_arch.pdf")


def find_2d_configuration():
    backend.set_benchmark(True)
    backend.set_deterministic(False)

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
    base_num_features = 32
    input_modalities = 4
    blocks_per_stage_encoder = (1, 3, 4, 6, 6, 6, 6)
    blocks_per_stage_decoder = (2, 2, 2, 2, 2, 2)
    feat_map_mult_on_downscale = 2
    num_classes = 5
    max_features = 512
    batch_size = 50

    unet = backend.to(FabiansPreActUNet(input_modalities, base_num_features, blocks_per_stage_encoder, feat_map_mult_on_downscale,
    pool_op_kernel_sizes, conv_op_kernel_sizes, get_default_network_config(2, dropout_p=None), num_classes,
    blocks_per_stage_decoder, True, False, max_features=max_features))

    scaler = backend.get_gradscaler()
    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)

    print(unet.compute_approx_vram_consumption(patch_size, base_num_features, max_features, input_modalities,
                                               num_classes, pool_op_kernel_sizes, blocks_per_stage_encoder,
                                               blocks_per_stage_decoder, feat_map_mult_on_downscale, batch_size))

    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    dummy_input = backend.to(torch.rand((batch_size, input_modalities, *patch_size)))
    dummy_gt = backend.to((torch.rand((batch_size, 1, *patch_size)) * num_classes).round().clamp_(0, num_classes-1)).long()

    for i in range(10):
        optimizer.zero_grad()

        with backend.autocast():
            skips = unet.encoder(dummy_input)
            print([i.shape for i in skips])
            output = unet.decoder(skips)[0]

            l = loss(output, dummy_gt)
            print(l.item())
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

    with backend.autocast():
        import hiddenlayer as hl
        g = hl.build_graph(unet, dummy_input, transforms=None)
        g.save("/home/fabian/test_arch.pdf")
