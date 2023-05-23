from typing import T
import torch
from torch.nn import Conv3d
from einops import rearrange, repeat
from torch import nn, einsum
from inspect import isfunction


class UNetEncoderS(nn.Module):

    def __init__(self, channels):
        super(UNetEncoderS, self).__init__()
        self.inc = (DoubleConv(channels, 16))
        self.down1 = (Down(16, 32, pooling=(1, 2, 2)))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        self.down4 = (Down(128, 256))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        skips = [x1, x2, x3, x4]
        return x5, skips


class UNetEncoderL(nn.Module):
    def __init__(self, channels):
        super(UNetEncoderL, self).__init__()
        self.inc = (DoubleConv(channels, 32))
        self.down1 = (Down(32, 64, pooling=(1, 2, 2)))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        skips = [x1, x2, x3, x4]
        return x5, skips


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = DoubleConv(in_channels, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class SegmentationHeadS(nn.Module):
    def __init__(self, in_features, segmentation_classes, do_ds):
        super(SegmentationHeadS, self).__init__()
        self.up_segmentation1 = (Up(in_features, 128,
                                    bilinear=False)
                                 )
        self.up_segmentation2 = (Up(128, 64, bilinear=False))
        self.up_segmentation3 = (Up(64, 32, bilinear=False))
        self.up_segmentation4 = (Up(32, 16, bilinear=False,
                                    pooling=(1, 2, 2)))
        self.outc_segmentation = (OutConv(16, segmentation_classes))

        self.do_ds = do_ds
        self.proj1 = Conv3d(256, segmentation_classes, 1)
        self.proj2 = Conv3d(128, segmentation_classes, 1)
        self.proj3 = Conv3d(64, segmentation_classes, 1)
        self.proj4 = Conv3d(32, segmentation_classes, 1)
        self.non_lin = lambda x: x  # torch.softmax(x, 1)

    def forward(self, x, skips):
        x1, x2, x3, x4 = skips
        if self.do_ds:
            outputs = []
            outputs.append(self.non_lin(self.proj1(x)))
        x = self.up_segmentation1(x, x4)
        if self.do_ds:
            outputs.append(self.non_lin(self.proj2(x)))
        x = self.up_segmentation2(x, x3)
        if self.do_ds:
            outputs.append(self.non_lin(self.proj3(x)))
        x = self.up_segmentation3(x, x2)
        if self.do_ds:
            outputs.append(self.non_lin(self.proj4(x)))
        x = self.up_segmentation4(x, x1)
        x = self.outc_segmentation(x)
        if self.do_ds:
            outputs.append(self.non_lin(x))
        if self.do_ds:
            return outputs[::-1]
        return self.non_lin(x)

    def eval(self: T) -> T:
        a = super(SegmentationHeadS, self).eval()
        self.do_ds = False
        return a

    def train(self: T, mode: bool = True) -> T:
        a = super(SegmentationHeadS, self).train()
        self.do_ds = True
        return a


class DoubleConv(nn.Module):

    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels,
                 mid_channels=None, dropout_rate: float = 0.):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3,
                      padding=1, bias=True),
            nn.Dropout3d(dropout_rate),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3,
                      padding=1, bias=True),
            nn.Dropout3d(dropout_rate),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pooling=(2,2,2)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(pooling),
            DoubleConv(in_channels, out_channels, dropout_rate=0.5)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, pooling=(2, 2, 2), bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=pooling, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=pooling, stride=pooling, bias=False)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SegmentationHeadL(nn.Module):
    def __init__(self, in_features, segmentation_classes, do_ds):
        super(SegmentationHeadL, self).__init__()
        self.up_segmentation1 = (Up(in_features, 256, bilinear=False))
        self.up_segmentation2 = (Up(256, 128, bilinear=False))
        self.up_segmentation3 = (Up(128, 64, bilinear=False))
        self.up_segmentation4 = (Up(64, 32, bilinear=False,
                                    pooling=(1, 2, 2)))
        self.outc_segmentation = (OutConv(32, segmentation_classes))

        self.do_ds = do_ds
        self.proj1 = Conv3d(512, segmentation_classes, 1)
        self.proj2 = Conv3d(256, segmentation_classes, 1)
        self.proj3 = Conv3d(128, segmentation_classes, 1)
        self.proj4 = Conv3d(64, segmentation_classes, 1)
        self.non_lin = lambda x: x  # torch.softmax(x, 1)

    def forward(self, x, skips):
        x1, x2, x3, x4 = skips
        if self.do_ds:
            outputs = []
            outputs.append(self.non_lin(self.proj1(x)))
        x = self.up_segmentation1(x, x4)
        if self.do_ds:
            outputs.append(self.non_lin(self.proj2(x)))
        x = self.up_segmentation2(x, x3)
        if self.do_ds:
            outputs.append(self.non_lin(self.proj3(x)))
        x = self.up_segmentation3(x, x2)
        if self.do_ds:
            outputs.append(self.non_lin(self.proj4(x)))
        x = self.up_segmentation4(x, x1)
        x = self.outc_segmentation(x)
        if self.do_ds:
            outputs.append(self.non_lin(x))
        if self.do_ds:
            return outputs[::-1]
        return self.non_lin(x)

    def eval(self: T) -> T:
        a = super(SegmentationHeadL, self).eval()
        self.do_ds = False
        return a

    def train(self: T, mode: bool = True) -> T:
        a = super(SegmentationHeadL, self).train()
        self.do_ds = True
        return a


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None,
                 heads=8, dim_head=64, dropout=0.):

        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Conv3d(query_dim, inner_dim,
                              kernel_size=1, bias=False)
        self.to_k = nn.Conv3d(context_dim, inner_dim,
                              kernel_size=1, bias=False)
        self.to_v = nn.Conv3d(context_dim, inner_dim,
                              kernel_size=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv3d(inner_dim, query_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        n = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, c, h, w, d = x.shape
        q, k, v = map(lambda t:
                      rearrange(t, 'b (n c) h w l -> (b n) c (h w l)',
                                n=n), (q, k, v))

        # force cast to fp32 to avoid overflowing
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b n) c (h w l) -> b (n c) h w l', n=n, h=h, w=w, l=d)
        out = self.to_out(out)
        return out


class UNetDeepSupervisionDoubleEncoder(nn.Module):
    def __init__(self, n_channels_1, n_channels_2,
                 n_classes_segmentation, deep_supervision=True,
                 encoder=UNetEncoderL,
                 segmentation_head=SegmentationHeadL):
        super(UNetDeepSupervisionDoubleEncoder, self).__init__()
        self.n_channels_1 = n_channels_1
        self.n_channels_2 = n_channels_2
        self.n_classes_segmentation = n_classes_segmentation
        self.nb_decoders = 4
        self.do_ds = deep_supervision
        self.deep_supervision = deep_supervision

        self.encoder1 = encoder(self.n_channels_1)
        self.encoder2 = encoder(self.n_channels_2)
        feature_size = 512 if encoder == UNetEncoderL else 256
        self.segmentation_head = segmentation_head(feature_size,
                                                   self.n_classes_segmentation,
                                                   self.do_ds)
        # self.CA = CrossAttention(query_dim=feature_size)

    def forward(self, x_in):
        x, y = x_in[:, 0:1, :, :, :], x_in[:, 1:2, :, :, :]
        features1, skips_1 = self.encoder1(x)
        features2, skips_2 = self.encoder2(y)
        x = torch.cat([features1, features2], dim=1)
        skip = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(skips_1, skips_2)]

        # self.CA(features1, context=features2)
        return self.segmentation_head(x, skip)

    def eval(self: T) -> T:
        super(UNetDeepSupervisionDoubleEncoder, self).eval()
        self.do_ds = False
        self.segmentation_head.eval()
        self.encoder1.eval()
        self.encoder2.eval()

    def train(self: T, mode: bool = True) -> T:
        super(UNetDeepSupervisionDoubleEncoder, self).train()
        self.do_ds = True
        self.encoder1.train()
        self.encoder2.train()
        self.segmentation_head.train()

