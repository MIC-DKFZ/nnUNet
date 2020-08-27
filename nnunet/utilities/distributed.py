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
from torch import distributed
from torch import autograd
from torch.nn.parallel import DistributedDataParallel as DDP


def print_if_rank0(*args):
    if distributed.get_rank() == 0:
        print(*args)


class awesome_allgather_function(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        world_size = distributed.get_world_size()
        # create a destination list for the allgather.  I'm assuming you're gathering from 3 workers.
        allgather_list = [torch.empty_like(input) for _ in range(world_size)]
        #if distributed.get_rank() == 0:
        #    import IPython;IPython.embed()
        distributed.all_gather(allgather_list, input)
        return torch.cat(allgather_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        #print_if_rank0("backward grad_output len", len(grad_output))
        #print_if_rank0("backward grad_output shape", grad_output.shape)
        grads_per_rank = grad_output.shape[0] // distributed.get_world_size()
        rank = distributed.get_rank()
        # We'll receive gradients for the entire catted forward output, so to mimic DataParallel,
        # return only the slice that corresponds to this process's input:
        sl = slice(rank * grads_per_rank, (rank + 1) * grads_per_rank)
        #print("worker", rank, "backward slice", sl)
        return grad_output[sl]


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
    c = DDP(c)
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
