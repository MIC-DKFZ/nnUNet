"""
The batch-chunked CE and Dice (nnunetv2/training/loss/chunked_losses.py).

They exist only to keep every nll_loss2d kernel launch below PyTorch's int32 indexing limit, and their whole
justification is that they compute the same thing as the stock losses. That is what these tests pin:

  * below the chunk budget nothing is chunked and the parent computation runs - bit-for-bit identical,
  * above it the dice is still bit-for-bit identical (its per-sample statistics reduce over spatial axes
    only, so chunking + concatenation reproduces the same tensors),
  * above it the CE differs only by floating point summation order.

The budget is lowered in the tests instead of building a 2**31 element tensor.
"""
import unittest

import torch

from nnunetv2.training.loss.chunked_losses import (
    ChunkedMemoryEfficientSoftDiceLoss,
    ChunkedRobustCrossEntropyLoss,
    DC_and_CE_loss_chunked,
)
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def make_batch(b=6, c=7, shape=(8, 9, 10), seed=0, ignore_label=None):
    torch.manual_seed(seed)
    logits = torch.randn((b, c, *shape), requires_grad=True)
    high = c + 1 if ignore_label is not None else c
    target = torch.randint(0, high, (b, 1, *shape))
    return logits, target


def value_and_grad(loss_fn, logits, target):
    logits = logits.detach().clone().requires_grad_(True)
    out = loss_fn(logits, target)
    out.backward()
    return out.detach(), logits.grad


class TestChunkedDice(unittest.TestCase):
    def _compare(self, batch_dice, do_bg, with_mask):
        logits, target = make_batch()
        mask = None
        if with_mask:
            mask = (torch.rand((logits.shape[0], 1, *logits.shape[2:])) > 0.3).float()

        kwargs = dict(apply_nonlin=softmax_helper_dim1, batch_dice=batch_dice, do_bg=do_bg, smooth=1e-5,
                      ddp=False)
        ref = MemoryEfficientSoftDiceLoss(**kwargs)
        chunked = ChunkedMemoryEfficientSoftDiceLoss(**kwargs)
        chunked.max_elements_per_chunk = logits[0].numel() * 2  # -> 2 samples per chunk, 3 chunks

        v_ref, g_ref = value_and_grad(lambda x, y: ref(x, y, mask), logits, target)
        v_chk, g_chk = value_and_grad(lambda x, y: chunked(x, y, mask), logits, target)
        self.assertTrue(torch.equal(v_ref, v_chk), f'{v_ref.item()} vs {v_chk.item()}')
        self.assertTrue(torch.equal(g_ref, g_chk))

    def test_bit_identical_when_chunked(self):
        for batch_dice in (True, False):
            for do_bg in (True, False):
                for with_mask in (True, False):
                    with self.subTest(batch_dice=batch_dice, do_bg=do_bg, mask=with_mask):
                        self._compare(batch_dice, do_bg, with_mask)

    def test_below_budget_runs_the_parent(self):
        logits, target = make_batch()
        kwargs = dict(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=1e-5, ddp=False)
        v_ref, g_ref = value_and_grad(MemoryEfficientSoftDiceLoss(**kwargs), logits, target)
        # default budget is 2**31, far above this batch -> no chunking at all
        v_chk, g_chk = value_and_grad(ChunkedMemoryEfficientSoftDiceLoss(**kwargs), logits, target)
        self.assertTrue(torch.equal(v_ref, v_chk))
        self.assertTrue(torch.equal(g_ref, g_chk))


class TestChunkedCE(unittest.TestCase):
    def test_matches_unchunked_up_to_summation_order(self):
        logits, target = make_batch()
        ref = RobustCrossEntropyLoss()
        chunked = ChunkedRobustCrossEntropyLoss()
        chunked.max_elements_per_chunk = logits[0].numel() * 2

        v_ref, g_ref = value_and_grad(ref, logits, target)
        v_chk, g_chk = value_and_grad(chunked, logits, target)
        self.assertTrue(torch.allclose(v_ref, v_chk, rtol=0, atol=1e-6), f'{v_ref.item()} vs {v_chk.item()}')
        self.assertTrue(torch.allclose(g_ref, g_chk, rtol=0, atol=1e-7))

    def test_ignore_index_is_honoured(self):
        """The chunked mean divides by the number of NON-ignored voxels, accumulated across chunks."""
        logits, target = make_batch(c=7, ignore_label=7)
        ref = RobustCrossEntropyLoss(ignore_index=7)
        chunked = ChunkedRobustCrossEntropyLoss(ignore_index=7)
        chunked.max_elements_per_chunk = logits[0].numel() * 2

        v_ref, g_ref = value_and_grad(ref, logits, target)
        v_chk, g_chk = value_and_grad(chunked, logits, target)
        self.assertTrue(torch.allclose(v_ref, v_chk, rtol=0, atol=1e-6), f'{v_ref.item()} vs {v_chk.item()}')
        self.assertTrue(torch.allclose(g_ref, g_chk, rtol=0, atol=1e-7))

    def test_non_default_reduction_falls_back_to_the_parent(self):
        logits, target = make_batch()
        chunked = ChunkedRobustCrossEntropyLoss(reduction='sum')
        chunked.max_elements_per_chunk = logits[0].numel() * 2
        v_ref, _ = value_and_grad(RobustCrossEntropyLoss(reduction='sum'), logits, target)
        v_chk, _ = value_and_grad(chunked, logits, target)
        self.assertTrue(torch.equal(v_ref, v_chk))


class TestChunkedCompoundLoss(unittest.TestCase):
    def _build(self, chunked, ignore_label=None, budget=None):
        dice_kwargs = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False, 'ddp': False}
        if chunked:
            loss = DC_and_CE_loss_chunked(dice_kwargs, {}, weight_ce=1, weight_dice=1,
                                          ignore_label=ignore_label, max_elements_per_chunk=budget)
        else:
            loss = DC_and_CE_loss(dice_kwargs, {}, weight_ce=1, weight_dice=1, ignore_label=ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)
        return loss

    def test_matches_stock_compound_loss(self):
        for ignore_label in (None, 7):
            with self.subTest(ignore_label=ignore_label):
                logits, target = make_batch(c=7, ignore_label=ignore_label)
                budget = logits[0].numel() * 2
                v_ref, g_ref = value_and_grad(self._build(False, ignore_label), logits, target)
                v_chk, g_chk = value_and_grad(self._build(True, ignore_label, budget), logits, target)
                self.assertTrue(torch.allclose(v_ref, v_chk, rtol=0, atol=1e-6),
                                f'{v_ref.item()} vs {v_chk.item()}')
                self.assertTrue(torch.allclose(g_ref, g_chk, rtol=0, atol=1e-7))

    def test_budget_reaches_both_components(self):
        budget = 12345
        loss = self._build(True, budget=budget)
        self.assertEqual(loss.ce.max_elements_per_chunk, budget)
        self.assertEqual(loss.dc.max_elements_per_chunk, budget)
        self.assertIsInstance(loss.ce, ChunkedRobustCrossEntropyLoss)
        self.assertIsInstance(loss.dc, ChunkedMemoryEfficientSoftDiceLoss)


if __name__ == '__main__':
    unittest.main()
