import os
import socket
from typing import Any, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist


class DDPTopology(NamedTuple):
    local_rank: int  # index of this process within its node -> this is the CUDA device index!
    global_rank: int  # index of this process within the entire job (across all nodes)
    world_size: int  # total number of processes in the job
    local_world_size: int  # number of processes on this node


def _first_env_int(*names: str) -> Optional[int]:
    for n in names:
        if n in os.environ:
            try:
                return int(os.environ[n])
            except ValueError:
                pass
    return None


def _topology_from_hostnames(global_rank: int, world_size: int) -> Tuple[int, int]:
    """
    (local_rank, local_world_size) derived by asking every rank which host it is on. Exact, but it costs a
    collective, so we only do this when the launcher did not tell us.
    """
    hostnames = [None] * world_size
    dist.all_gather_object(hostnames, socket.gethostname())
    mine = hostnames[global_rank]
    ranks_on_my_host = [i for i, h in enumerate(hostnames) if h == mine]
    return ranks_on_my_host.index(global_rank), len(ranks_on_my_host)


def get_ddp_topology() -> DDPTopology:
    """
    Where this process sits in the DDP job. Must only be called once a process group exists.

    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE and LOCAL_WORLD_SIZE, and `nnUNetv2_train -num_gpus X` sets the
    same four variables before spawning its workers, so for both of nnU-Net's own launch paths these are simply
    read back. They are the authoritative source: local_rank in particular cannot be read off the process group,
    because torch.distributed has no concept of nodes.

    Third party launchers that create the process group themselves (MPI, a hand-rolled init_method, an embedding
    script) may not set them. For those we fall back to the common scheduler variables, and finally to asking the
    other ranks which host they are on, rather than raising KeyError.
    """
    assert dist.is_available() and dist.is_initialized(), \
        'get_ddp_topology() requires an initialized process group'

    global_rank = _first_env_int('RANK')
    if global_rank is None:
        global_rank = dist.get_rank()

    world_size = _first_env_int('WORLD_SIZE')
    if world_size is None:
        world_size = dist.get_world_size()

    # SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK cover srun and mpirun launches
    local_rank = _first_env_int('LOCAL_RANK', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK',
                                'MV2_COMM_WORLD_LOCAL_RANK')
    local_world_size = _first_env_int('LOCAL_WORLD_SIZE', 'SLURM_NTASKS_PER_NODE', 'OMPI_COMM_WORLD_LOCAL_SIZE',
                                      'MV2_COMM_WORLD_LOCAL_SIZE')

    if local_rank is None or local_world_size is None:
        derived_local_rank, derived_local_world_size = _topology_from_hostnames(global_rank, world_size)
        if local_rank is None:
            local_rank = derived_local_rank
        if local_world_size is None:
            local_world_size = derived_local_world_size
        print(f'LOCAL_RANK/LOCAL_WORLD_SIZE were not set, so your process group was not created by torchrun or by '
              f'nnUNetv2_train -num_gpus. Derived local_rank={local_rank} and local_world_size={local_world_size} '
              f'by comparing hostnames across the {world_size} ranks. Set LOCAL_RANK yourself if your launcher '
              f'assigns GPUs differently.')

    return DDPTopology(local_rank=local_rank, global_rank=global_rank, world_size=world_size,
                       local_world_size=local_world_size)


class AllReduceGrad(torch.autograd.Function):
    """
    Differentiable SUM all-reduce: every rank gets the sum of all ranks' tensors, and in the backward pass every
    rank gets the sum of all ranks' gradients (which is what dL/dx_rank is when the tensor is summed into a loss
    that every rank computes).

    This is what torch.distributed.nn.functional.all_reduce does, but that one is deprecated as of torch 2.13 in
    favour of the private torch.distributed._functional_collectives, and we would rather not depend on either.
    It is also equivalent to the old AllGatherGrad.apply(x).sum(0) while moving world_size times less data.
    """
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor,
                group: Optional["torch.distributed.ProcessGroup"] = None) -> torch.Tensor:
        ctx.group = group
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = grad_output.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None
