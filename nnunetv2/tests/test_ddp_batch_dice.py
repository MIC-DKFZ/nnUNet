"""
Batch Dice under DDP when the ranks hold different numbers of samples.

_set_batch_size_and_oversample gives the first ranks one sample more when the global batch does not divide by the
number of GPUs (e.g. 3 + 2 for a batch of 5 on 2 GPUs). MemoryEfficientSoftDiceLoss used to all-reduce its per-sample
(b, c) statistics, whose b then differs across ranks: gloo aborts the job, NCCL hangs. The
statistics are now summed over each rank's samples first, so every rank all-reduces a (c,) tensor.

These tests run two CPU processes with the gloo backend and compare against a single process that sees the whole
batch. They join with a timeout, so a regression fails instead of hanging the test suite.
"""
import multiprocessing
import os
import socket
import traceback
import unittest

import torch

from nnunetv2.training.loss.chunked_losses import ChunkedMemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

TIMEOUT_S = 120
LOSS_KWARGS = dict(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=1e-5)


def _make_batch(batch_size: int, num_classes: int = 4, shape=(6, 7, 8)):
    torch.manual_seed(0)
    logits = torch.randn((batch_size, num_classes, *shape))
    target = torch.randint(0, num_classes, (batch_size, 1, *shape))
    return logits, target


def _make_loss(chunked: bool, ddp: bool):
    if chunked:
        loss = ChunkedMemoryEfficientSoftDiceLoss(ddp=ddp, **LOSS_KWARGS)
        loss.max_elements_per_chunk = 1  # one sample per chunk, so the chunked path really runs
        return loss
    return MemoryEfficientSoftDiceLoss(ddp=ddp, **LOSS_KWARGS)


def _rank_slices(sizes):
    starts = [sum(sizes[:r]) for r in range(len(sizes))]
    return [slice(s, s + n) for s, n in zip(starts, sizes)]


def _worker(rank: int, sizes, port: int, chunked: bool, results):
    try:
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group('gloo', rank=rank, world_size=len(sizes))
        logits, target = _make_batch(sum(sizes))
        sl = _rank_slices(sizes)[rank]
        x = logits[sl].clone().requires_grad_(True)
        loss = _make_loss(chunked, ddp=True)(x, target[sl])
        loss.backward()
        results.put((rank, loss.item(), x.grad.numpy(), None))  # numpy: no shared memory handle to a process that exits
        dist.destroy_process_group()
    except Exception:
        results.put((rank, None, None, traceback.format_exc()))


def _run_ddp(sizes, chunked: bool):
    ctx = multiprocessing.get_context('spawn')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    results = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, sizes, port, chunked, results)) for r in range(len(sizes))]
    for p in procs:
        p.start()
    out = {}
    try:
        for _ in procs:
            rank, loss, grad, err = results.get(timeout=TIMEOUT_S)
            if err is not None:
                raise AssertionError(f'rank {rank} failed:\n{err}')
            out[rank] = (loss, torch.from_numpy(grad))
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
    return out


class TestDDPBatchDice(unittest.TestCase):
    def _check(self, sizes, chunked: bool):
        out = _run_ddp(sizes, chunked)

        logits, target = _make_batch(sum(sizes))
        x = logits.clone().requires_grad_(True)
        ref_loss = _make_loss(chunked, ddp=False)(x, target)
        ref_loss.backward()

        for rank, sl in enumerate(_rank_slices(sizes)):
            loss, grad = out[rank]
            self.assertAlmostEqual(loss, ref_loss.item(), places=6)
            # every rank computes the same global loss and AllReduceGrad sums its gradient over the ranks; DDP then
            # averages the parameter gradients over the ranks. Divided by world_size, each rank's gradient is the
            # single-process gradient of its samples.
            torch.testing.assert_close(grad / len(sizes), x.grad[sl], rtol=1e-5, atol=1e-7)

    def test_uneven_split(self):
        for chunked in (False, True):
            for sizes in ((3, 2), (2, 1)):
                with self.subTest(chunked=chunked, sizes=sizes):
                    self._check(sizes, chunked)

    def test_even_split(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                self._check((2, 2), chunked)


if __name__ == '__main__':
    unittest.main()
