from nnunet.utilities.tensor_utilities import mean_tensor
from torch import nn
import torch
from torch.nn.parameter import Parameter
import torch.jit


class FRN3D(nn.Module):
    def __init__(self, num_features: int, eps=1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.weight = Parameter(torch.ones(1, num_features, 1, 1, 1), True)
        self.bias = Parameter(torch.zeros(1, num_features, 1, 1, 1), True)
        self.tau = Parameter(torch.zeros(1, num_features, 1, 1, 1), True)

    def forward(self, x: torch.Tensor):
        x = x * torch.rsqrt(mean_tensor(x * x, [2, 3, 4], keepdim=True) + self.eps)

        return torch.max(self.weight * x + self.bias, self.tau)


if __name__ == "__main__":
    tmp = torch.rand((3, 32, 16, 16, 16))

    frn = FRN3D(32)

    out = frn(tmp)
