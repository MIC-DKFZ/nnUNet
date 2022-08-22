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
from typing import Any, Optional, Tuple

import torch
from torch import distributed
from torch import autograd
from torch.nn.parallel import DistributedDataParallel as DDP


def print_if_rank0(*args):
    if distributed.get_rank() == 0:
        print(*args)


class awesome_allgather_function(autograd.Function):
    """
    @mcarili (Michael Carilli) from Nvidia wrote this code for me a long time ago:
    https://github.com/NVIDIA/apex/issues/162
    Full credit to him! Amazing dude!
    """
    @staticmethod
    def forward(ctx, input):
        world_size = distributed.get_world_size()
        # create a destination list for the allgather
        allgather_list = [torch.empty_like(input) for _ in range(world_size)]
        distributed.all_gather(allgather_list, input)
        return torch.cat(allgather_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grads_per_rank = grad_output.shape[0] // distributed.get_world_size()
        rank = distributed.get_rank()
        # We'll receive gradients for the entire catted forward output, so to mimic DataParallel,
        # return only the slice that corresponds to this process's input:
        sl = slice(rank * grads_per_rank, (rank + 1) * grads_per_rank)
        return grad_output[sl]


class AllGatherGrad(torch.autograd.Function):
    # stolen from pytorch lightning
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> torch.Tensor:
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


def all_gather_ddp_if_available(
    tensor: torch.Tensor, group: Optional["torch.distributed.ProcessGroup"] = None, sync_grads: bool = False
) -> torch.Tensor:
    """Function to gather a tensor from several distributed processes.

    Args:
        tensor: tensor of shape (batch, ...)
        group: the process group to gather results from. Defaults to all processes (world)
        sync_grads: flag that allows users to synchronize gradients for all_gather op

    Return:
        A tensor of shape (world_size, batch, ...)
    """
    # stolen from pytorch lightning
    group = group if group is not None else torch.distributed.group.WORLD
    if distributed_available():
        if sync_grads:
            return AllGatherGrad.apply(tensor, group)
        with torch.no_grad():
            return AllGatherGrad.apply(tensor, group)
    return tensor


def distributed_available() -> bool:
    # stolen from pytorch lightning
    return torch.distributed.is_available() and torch.distributed.is_initialized() or tpu_distributed()


if __name__ == "__main__":
    import torch.distributed as dist
    import argparse
    from torch import nn
    from torch.optim import Adam

    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument("--local_rank", type=int)
    args = argumentparser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rnd = torch.rand((5, 2)).cuda()

    rnd_gathered = awesome_allgather_function.apply(rnd)
    print("gathering random tensors\nbefore\b", rnd, "\nafter\n", rnd_gathered)

    # so far this works as expected
    print("now running a DDP model")
    c = nn.Conv2d(2, 3, 3, 1, 1, 1, 1, True).cuda()
    c = DDP(c, args.local_rank)
    opt = Adam(c.parameters())

    bs = 5
    if dist.get_rank() == 0:
        bs = 4
    inp = torch.rand((bs, 2, 5, 5)).cuda()

    out = c(inp)
    print("output_shape", out.shape)

    out_gathered = awesome_allgather_function.apply(out)
    print("output_shape_after_gather", out_gathered.shape)
    # this also works

    loss = out_gathered.sum()
    loss.backward()
    opt.step()
