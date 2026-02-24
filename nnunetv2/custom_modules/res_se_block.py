from typing import Callable, Optional, Type
import torch 
from torch import nn

class SE3D(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        act: Optional[Callable[[], nn.Module]] = None,
        gate: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0, got {reduction}")

        act = act or (lambda: nn.ReLU(inplace=True))
        gate = gate or (lambda: nn.Sigmoid())

        hidden = max(1, channels // reduction)

        self.pool = nn.AdaptiveAvgPool3d(1) 
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.act = act()
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)
        self.gate = gate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w

class ResidualSEBlock3D(nn.Module):
   
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        bias: bool = False,
        conv_op: Type[nn.Module] = nn.Conv3d,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_kwargs: Optional[dict] = None,
        nonlin: Optional[Callable[[], nn.Module]] = None,
        se_reduction: int = 16,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2
        norm_kwargs = norm_kwargs or {}
        nonlin = nonlin or (lambda: nn.LeakyReLU(inplace=True))

        self.conv1 = conv_op(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.norm1 = norm_op(out_ch, **norm_kwargs)
        self.act1 = nonlin()

        self.conv2 = conv_op(
            out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=bias
        )
        self.norm2 = norm_op(out_ch, **norm_kwargs)

        self.se = SE3D(out_ch, reduction=se_reduction)

        # projection for residual if shape mismatch
        needs_proj = (in_ch != out_ch) or (stride != 1)
        if needs_proj:
            self.proj = nn.Sequential(
                conv_op(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=bias),
                norm_op(out_ch, **norm_kwargs),
            )
        else:
            self.proj = None

        self.act_out = nonlin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.se(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act_out(out)
        return out