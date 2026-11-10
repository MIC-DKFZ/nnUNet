import unittest

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from nnunetv2.inference.sliding_window_prediction import compute_gaussian


def _reference_compute_gaussian(tile_size, sigma_scale=1. / 8, value_scaling_factor=1, dtype=torch.float16,
                                device=torch.device('cpu')):
    """compute_gaussian as it was before it was built from 1d responses: scipy's 3d filter on a centred delta"""
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map /= (np.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
    return gaussian_importance_map


class TestComputeGaussian(unittest.TestCase):
    TILE_SIZES = (
        (64, 64, 64),
        (128, 112, 80),
        (33, 47, 21),     # odd sizes
        (160, 8, 96),     # one short axis
        (320, 256),       # 2d
        (17, 9),
        (5, 1, 7),        # an axis of length 1
    )

    def test_bit_identical_to_3d_filter(self):
        for tile_size in self.TILE_SIZES:
            for sigma_scale in (1. / 8, 1. / 4, 1. / 64, 0.):
                # float64 is the strictest comparison, float16 with value_scaling_factor 10 is what the predictor uses
                for value_scaling_factor, dtype in ((1, torch.float64), (10, torch.float16)):
                    with self.subTest(tile_size=tile_size, sigma_scale=sigma_scale, dtype=dtype):
                        expected = _reference_compute_gaussian(tile_size, sigma_scale, value_scaling_factor, dtype)
                        # __wrapped__: bypass the lru_cache so every combination is actually computed
                        got = compute_gaussian.__wrapped__(tile_size, sigma_scale, value_scaling_factor, dtype,
                                                           torch.device('cpu'))
                        self.assertEqual(got.dtype, expected.dtype)
                        self.assertEqual(tuple(got.shape), tuple(tile_size))
                        self.assertTrue(torch.equal(got, expected))

    def test_list_tile_size(self):
        # the predictor passes a tuple (lru_cache needs hashable arguments), but the signature also allows lists
        got = compute_gaussian.__wrapped__([24, 20, 16], 1. / 8, 10, torch.float16, torch.device('cpu'))
        expected = _reference_compute_gaussian((24, 20, 16), 1. / 8, 10, torch.float16)
        self.assertTrue(torch.equal(got, expected))


if __name__ == '__main__':
    unittest.main()
