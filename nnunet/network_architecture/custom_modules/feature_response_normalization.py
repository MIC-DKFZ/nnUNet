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
