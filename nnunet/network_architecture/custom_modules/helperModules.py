from torch import nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class MyGroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True, num_groups=8):
        super(MyGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
