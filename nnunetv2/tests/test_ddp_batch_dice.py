"""
Batch Dice under DDP when the ranks hold different numbers of samples.

_set_batch_size_and_oversample gives the first ranks one sample more when the global batch does not divide by the
number of GPUs (e.g. 3 + 2 for a batch of 5 on 2 GPUs). MemoryEfficientSoftDiceLoss used to all-reduce its per-sample
(b, c) statistics, whose b then differs across ranks: gloo aborts the job, NCCL hangs. The
statistics are now summed over each rank's samples first, so every rank all-reduces a (c,) tensor.

Two CPU processes with the gloo backend run all cases in one process group (spawning is what takes the time), and
each case is compared against a single process that sees the whole batch. The even splits run first, so they are
checked even if an uneven split breaks the group. The processes are joined with a timeout, so a regression fails
instead of hanging the test suite.
"""
import multiprocessing
import os
import queue
import shutil
import tempfile
import time
import traceback
import unittest

import torch

from nnunetv2.training.loss.chunked_losses import ChunkedMemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

TIMEOUT_S = 120
WORLD_SIZE = 2
LOSS_KWARGS = dict(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=1e-5)
# (samples per rank, chunked loss); even splits first, see the module docstring
EVEN = [((2, 2), False), ((2, 2), True)]
UNEVEN = [((3, 2), False), ((2, 1), False), ((3, 2), True), ((2, 1), True)]
CASES = EVEN + UNEVEN


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


def _worker(rank: int, init_file: str, results):
    import torch.distributed as dist
    try:
        # file:// rendezvous: no free port to find, so nothing another process on the machine can take
        dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=WORLD_SIZE)
    except Exception:
        results.put((None, rank, None, None, traceback.format_exc()))
        return
    for i, (sizes, chunked) in enumerate(CASES):
        try:
            logits, target = _make_batch(sum(sizes))
            sl = _rank_slices(sizes)[rank]
            x = logits[sl].clone().requires_grad_(True)
            loss = _make_loss(chunked, ddp=True)(x, target[sl])
            loss.backward()
            # numpy: no shared memory handle to a process that exits
            results.put((i, rank, loss.item(), x.grad.numpy(), None))
        except Exception:
            results.put((i, rank, None, None, traceback.format_exc()))
            return  # the process group may be broken now; the remaining cases are reported as not run
    dist.destroy_process_group()


def _run_all_cases():
    """{case index: {rank: (loss, grad) or error string}}, for the cases that produced a result."""
    tmp = tempfile.mkdtemp()
    ctx = multiprocessing.get_context('spawn')
    results = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, os.path.join(tmp, 'init'), results)) for r in range(WORLD_SIZE)]
    for p in procs:
        p.start()
    out = {}
    try:
        deadline = time.time() + TIMEOUT_S
        expected = len(CASES) * WORLD_SIZE
        received = 0
        while received < expected and time.time() < deadline:
            try:
                i, rank, loss, grad, err = results.get(timeout=1)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    break  # a rank died without reporting (gloo aborts the process on a size mismatch)
                continue
            received += 1
            if i is None:
                raise AssertionError(f'rank {rank} could not join the process group:\n{err}')
            out.setdefault(i, {})[rank] = err if err is not None else (loss, torch.from_numpy(grad))
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
        shutil.rmtree(tmp, ignore_errors=True)
    return out


class TestDDPBatchDice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = _run_all_cases()

    def _check(self, case_index: int):
        sizes, chunked = CASES[case_index]
        per_rank = self.results.get(case_index, {})
        for rank in range(WORLD_SIZE):
            if rank not in per_rank:
                self.fail(f'rank {rank} returned no result (it died, timed out or stopped at an earlier case)')
            if isinstance(per_rank[rank], str):
                self.fail(f'rank {rank} failed:\n{per_rank[rank]}')

        logits, target = _make_batch(sum(sizes))
        x = logits.clone().requires_grad_(True)
        ref_loss = _make_loss(chunked, ddp=False)(x, target)
        ref_loss.backward()

        for rank, sl in enumerate(_rank_slices(sizes)):
            loss, grad = per_rank[rank]
            self.assertAlmostEqual(loss, ref_loss.item(), places=6)
            # every rank computes the same global loss and AllReduceGrad sums its gradient over the ranks; DDP then
            # averages the parameter gradients over the ranks. Divided by world_size, each rank's gradient is the
            # single-process gradient of its samples.
            torch.testing.assert_close(grad / WORLD_SIZE, x.grad[sl], rtol=1e-5, atol=1e-7)

    def test_even_split(self):
        for i in range(len(EVEN)):
            sizes, chunked = CASES[i]
            with self.subTest(sizes=sizes, chunked=chunked):
                self._check(i)

    def test_uneven_split(self):
        for i in range(len(EVEN), len(CASES)):
            sizes, chunked = CASES[i]
            with self.subTest(sizes=sizes, chunked=chunked):
                self._check(i)


if __name__ == '__main__':
    unittest.main()
