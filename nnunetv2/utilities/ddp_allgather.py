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
import warnings
from typing import Any, Optional, Tuple

import torch


class AllGatherGrad(torch.autograd.Function):
    """
    DEPRECATED. nnU-Net now uses AllReduceGrad (nnunetv2.utilities.ddp), which computes the same thing (forward
    and backward) while moving world_size times less data: AllGatherGrad.apply(x).sum(0) is AllReduceGrad.apply(x).
    Kept because custom trainers and losses outside this repository import it. It will be removed in a future
    release.
    """
    # stolen from pytorch lightning
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> torch.Tensor:
        warnings.warn('AllGatherGrad is deprecated and will be removed in a future version of nnU-Net. If you '
                      'only sum over the gathered dimension, as nnU-Net\'s losses do, replace '
                      'AllGatherGrad.apply(x).sum(0) with nnunetv2.utilities.ddp.AllReduceGrad.apply(x).',
                      DeprecationWarning, stacklevel=2)
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None
