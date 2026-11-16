"""
The online (green line) evaluation in nnUNetTrainer.validation_step.

For softmax training that step no longer builds a (b, num_classes, *patch_size) one hot to hand to
get_tp_fp_fn_tn - at 118 classes / 192^3 / bs2 the one hot alone is ~3.3 GB and it was the VRAM ceiling of
the whole validation step on the many-class datasets. It accumulates a (num_classes, num_classes) confusion
matrix over (prediction, target) instead.

These tests pin that get_tp_fp_fn_from_argmax is EXACTLY the old computation, because "equivalent" is the entire
justification for the swap. The reference is always the one hot path the trainer used before, run on the
same argmax.
"""
import unittest

import numpy as np
import torch

from nnunetv2.training.loss.dice import get_tp_fp_fn_from_argmax, get_tp_fp_fn_tn


def reference_tp_fp_fn(output_seg: torch.Tensor, target: torch.Tensor, num_classes: int,
                       mask: torch.Tensor = None):
    """The one hot + get_tp_fp_fn_tn path, i.e. what nnUNetTrainer.validation_step used to do."""
    onehot = torch.zeros((output_seg.shape[0], num_classes, *output_seg.shape[2:]), dtype=torch.float16)
    onehot.scatter_(1, output_seg, 1)
    axes = [0] + list(range(2, onehot.ndim))
    tp, fp, fn, _ = get_tp_fp_fn_tn(onehot, target, axes=axes, mask=mask)
    return tp, fp, fn


class TestOnlineEvaluation(unittest.TestCase):
    def assert_same(self, output_seg, target, num_classes, mask=None):
        tp_r, fp_r, fn_r = reference_tp_fp_fn(output_seg, target, num_classes, mask)
        tp_c, fp_c, fn_c = get_tp_fp_fn_from_argmax(output_seg, target, num_classes, mask=mask)
        # the reference returns float, the confusion matrix returns int64 - the counts must match exactly
        self.assertTrue(torch.equal(tp_r.long(), tp_c), f'tp: {tp_r.tolist()} vs {tp_c.tolist()}')
        self.assertTrue(torch.equal(fp_r.long(), fp_c), f'fp: {fp_r.tolist()} vs {fp_c.tolist()}')
        self.assertTrue(torch.equal(fn_r.long(), fn_c), f'fn: {fn_r.tolist()} vs {fn_c.tolist()}')
        return tp_c, fp_c, fn_c

    def test_random_agrees_with_one_hot(self):
        torch.manual_seed(1234)
        for num_classes in (2, 5, 37):
            for shape in ((2, 1, 8, 9, 10), (1, 1, 16, 16, 16), (3, 1, 12, 5, 7)):
                output_seg = torch.randint(0, num_classes, shape)
                target = torch.randint(0, num_classes, shape)
                with self.subTest(num_classes=num_classes, shape=shape):
                    self.assert_same(output_seg, target, num_classes)

    def test_2d(self):
        torch.manual_seed(0)
        output_seg = torch.randint(0, 6, (4, 1, 24, 24))
        target = torch.randint(0, 6, (4, 1, 24, 24))
        self.assert_same(output_seg, target, 6)

    def test_fp_and_fn_are_not_swapped(self):
        """
        Dice is symmetric in fp and fn, so swapping them is invisible in the green line and only shows up
        when anyone reads the raw counts. Pin the orientation with an asymmetric case.
        """
        # everything is target class 1; the network says class 2 everywhere. So for class 1: only fn,
        # for class 2: only fp.
        target = torch.ones((1, 1, 4, 4, 4), dtype=torch.long)
        output_seg = torch.full((1, 1, 4, 4, 4), 2, dtype=torch.long)
        tp, fp, fn = self.assert_same(output_seg, target, 3)
        self.assertEqual(fn[1].item(), 64)
        self.assertEqual(fp[1].item(), 0)
        self.assertEqual(fp[2].item(), 64)
        self.assertEqual(fn[2].item(), 0)

    def test_label_present_in_target_but_never_predicted(self):
        """
        A class the network missed entirely must get fn > 0. Counting only over torch.unique(output_seg) would
        give it fn = 0, and because on_validation_epoch_end sums tp/fp/fn over the whole epoch before forming the
        Dice, that would inflate the green line for any class that was missed in some batches and hit in others.
        """
        target = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
        target[..., :2] = 3  # 32 voxels of class 3 in the target
        output_seg = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)  # class 3 never predicted
        tp, fp, fn = self.assert_same(output_seg, target, 4)
        self.assertEqual(tp[3].item(), 0)
        self.assertEqual(fp[3].item(), 0)
        self.assertEqual(fn[3].item(), 32)

    def test_class_absent_from_both_stays_absent(self):
        """A class in neither prediction nor target must give tp = fp = fn = 0, so the Dice is 0/0 = nan
        and np.nanmean drops it. Adding an epsilon to the denominator instead would score it 0."""
        target = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
        output_seg = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
        tp, fp, fn = self.assert_same(output_seg, target, 5)
        for label in (1, 2, 3, 4):
            self.assertEqual((tp[label].item(), fp[label].item(), fn[label].item()), (0, 0, 0))
        with np.errstate(invalid='ignore', divide='ignore'):
            dice = [2 * i / (2 * i + j + k) for i, j, k in zip(tp.numpy(), fp.numpy(), fn.numpy())]
        self.assertTrue(all(np.isnan(dice[label]) for label in (1, 2, 3, 4)))

    def test_ignore_label_is_respected(self):
        """
        Ignored voxels must not be counted. Without the mask they would count as background, since
        validation_step zeroes them in the target first. Here the ignored region is deliberately filled with a
        wrong prediction - if the mask is dropped it shows up as fp.
        """
        num_classes = 3
        ignore_label = num_classes  # this is how nnU-Net numbers it
        target = torch.ones((1, 1, 4, 4, 4), dtype=torch.long)
        target[..., :2] = ignore_label
        output_seg = torch.full((1, 1, 4, 4, 4), 2, dtype=torch.long)  # wrong everywhere, incl. the ignored half

        # exactly what validation_step does before the counting
        mask = (target != ignore_label).float()
        target = target.clone()
        target[target == ignore_label] = 0

        tp, fp, fn = self.assert_same(output_seg, target, num_classes, mask=mask)
        # only the 32 annotated voxels may be counted
        self.assertEqual(int(tp.sum() + fp.sum()), 32)
        self.assertEqual(fp[2].item(), 32)
        self.assertEqual(fn[1].item(), 32)

        # and the mask must actually be doing the work
        tp_unmasked, fp_unmasked, _ = get_tp_fp_fn_from_argmax(output_seg, target, num_classes, mask=None)
        self.assertEqual(fp_unmasked[2].item(), 64)


if __name__ == '__main__':
    unittest.main()
