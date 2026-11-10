import unittest

import numpy as np
import torch

from nnunetv2.utilities.label_handling.label_handling import LabelManager, argmax_first_axis


class TestArgmaxFirstAxis(unittest.TestCase):
    def _check(self, x, **kwargs):
        expected = x.argmax(0)
        got = argmax_first_axis(x, **kwargs)
        self.assertEqual(got.dtype, expected.dtype)
        self.assertEqual(got.shape, expected.shape)
        self.assertTrue(np.array_equal(got, expected))

    def test_same_as_numpy(self):
        rng = np.random.default_rng(0)
        for shape in ((11, 17, 13, 9), (3, 40, 31), (105, 6, 5, 4), (2, 1, 7, 7), (4, 33)):
            for dtype in (np.float16, np.float32, np.float64):
                # few distinct values, so that ties are frequent
                x = rng.integers(-4, 5, size=shape).astype(dtype) / 2
                for slab in (1, 7, 100, 2 ** 24):
                    with self.subTest(shape=shape, dtype=dtype, slab=slab):
                        self._check(x, max_elements_per_slab=slab)

    def test_special_values(self):
        x = np.zeros((4, 3, 5), dtype=np.float16)
        x[1, 0, 0] = np.inf
        x[2, 0, 0] = np.inf                      # tie of infs: first one
        x[:, 1, 1] = -np.inf                     # all -inf: 0
        x[2, 2, 2] = np.nan                      # NaN wins in numpy's argmax
        x[3, 2, 3] = np.nan
        x[1, 2, 3] = np.nan                      # first NaN
        x[0, 0, 4] = -0.
        x[1, 0, 4] = 0.                          # -0 == 0: first one
        x[3, 1, 2] = 65504                       # float16 max
        x[0, 2, 0] = np.float16(6e-8)            # subnormal
        for slab in (1, 4, 2 ** 24):
            self._check(x, max_elements_per_slab=slab)

    def test_non_contiguous(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(20, 6, 18, 10)).astype(np.float16)
        for view in (x[::2], x[:, :, ::3], x.transpose(0, 3, 1, 2), x[:, ::-1]):
            self._check(view, max_elements_per_slab=50)

    def test_label_manager_uses_it(self):
        rng = np.random.default_rng(2)
        lm = LabelManager({'background': 0, 'a': 1, 'b': 2}, None)
        logits = rng.normal(size=(3, 12, 10, 8)).astype(np.float16)
        self.assertTrue(np.array_equal(lm.convert_logits_to_segmentation(logits), logits.argmax(0)))
        seg = lm.convert_logits_to_segmentation(torch.from_numpy(logits))
        self.assertTrue(isinstance(seg, torch.Tensor))
        self.assertTrue(np.array_equal(seg.numpy(), logits.argmax(0)))


if __name__ == '__main__':
    unittest.main()
