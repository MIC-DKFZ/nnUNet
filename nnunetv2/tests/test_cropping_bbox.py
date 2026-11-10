import unittest

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice

from nnunetv2.preprocessing.cropping.cropping import create_nonzero_mask, crop_to_nonzero, get_nonzero_bbox


def _reference_crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """crop_to_nonzero as it was: binary_fill_holes on the whole image, then acvl's box of the filled mask"""
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox


def _projection_bbox(data):
    """box of the voxels nonzero in any channel from per-axis projections (also for 2d, which acvl does not take)"""
    nonzero = np.any(data != 0, axis=0)
    bbox = []
    for ax in range(nonzero.ndim):
        idx = np.flatnonzero(nonzero.any(axis=tuple(a for a in range(nonzero.ndim) if a != ax)))
        bbox.append([int(idx[0]), int(idx[-1]) + 1] if idx.size else [0, nonzero.shape[ax]])
    return bbox


def _random_images(rng, n, n_channels_choices=(1, 2, 3), max_size=24):
    """
    Nonzero inside a box that touches a random subset of the image borders. Inside: single zero voxels and sometimes
    an enclosed zero cavity (both filled by binary_fill_holes) or a zero plane across the whole image (not filled).
    """
    for _ in range(n):
        n_channels = int(rng.choice(n_channels_choices))
        shape = tuple(int(i) for i in rng.integers(1, max_size, size=3))
        lo = [int(rng.integers(0, s // 3 + 1)) if rng.random() < 0.7 else 0 for s in shape]
        hi = [int(rng.integers(max(a + 1, 2 * s // 3), s + 1)) if rng.random() < 0.7 else s for s, a in zip(shape, lo)]
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        data = np.zeros((n_channels, *shape), dtype=np.float32)
        for c in range(n_channels):
            values = rng.normal(20, 40, size=shape).astype(np.float32)
            values[rng.random(shape) < rng.choice([0.05, 0.7, 0.99])] = 0
            data[c][box] = values[box]
        r = rng.random()
        if r < 0.25:
            data[:, shape[0] // 2] = 0
        elif r < 0.6:
            data[:, shape[0] // 3: shape[0] // 2, shape[1] // 3: shape[1] // 2, shape[2] // 3: shape[2] // 2] = 0
        yield data


def _views(data):
    """the array and the transposed views preprocessing passes (transpose_forward), so other axes are outermost"""
    yield data
    yield data.transpose(0, 3, 1, 2)
    yield data.transpose(0, 2, 3, 1)


class TestGetNonzeroBbox(unittest.TestCase):
    def test_box_of_the_filled_nonzero_mask(self):
        rng = np.random.default_rng(5)
        for data in _random_images(rng, 150):
            for view in _views(data):
                with self.subTest(shape=view.shape, strides=view.strides):
                    self.assertEqual(get_nonzero_bbox(view), get_bbox_from_mask(create_nonzero_mask(view)))

    def test_singleton_axes_and_2d(self):
        rng = np.random.default_rng(6)
        for shape in ((1, 30, 30), (25, 1, 9), (16, 16, 1)):
            data = np.zeros((2, *shape), dtype=np.float32)
            data[1][rng.random(shape) < 0.1] = 1
            for view in _views(data):
                self.assertEqual(get_nonzero_bbox(view), _projection_bbox(view))
        data = np.zeros((2, 30, 20), dtype=np.float32)
        data[1, 4:9, 11:13] = -1.5
        data[0, 20, 3] = np.nan
        self.assertEqual(get_nonzero_bbox(data), [[4, 21], [3, 13]])
        self.assertEqual(get_nonzero_bbox(data.transpose(0, 2, 1)), [[3, 13], [4, 21]])

    def test_all_zero_and_full(self):
        self.assertEqual(get_nonzero_bbox(np.zeros((2, 5, 6, 7), dtype=np.float32)), [[0, 5], [0, 6], [0, 7]])
        self.assertEqual(get_nonzero_bbox(np.ones((1, 5, 6, 7), dtype=np.float32)), [[0, 5], [0, 6], [0, 7]])
        # nonzero on every face of the first-axis slab but not in its corners: still the full extent
        frame = np.zeros((1, 20, 30, 40), dtype=np.float32)
        frame[0, :, 0, 20] = frame[0, :, -1, 20] = frame[0, :, 15, 0] = frame[0, :, 15, -1] = 1
        self.assertEqual(get_nonzero_bbox(frame), [[0, 20], [0, 30], [0, 40]])

    def test_returns_python_ints(self):
        data = np.zeros((1, 9, 8, 7), dtype=np.float32)
        data[0, 2:5, 3, 1:4] = 1
        for view in _views(data):
            for start, end in get_nonzero_bbox(view):
                self.assertIs(type(start), int)
                self.assertIs(type(end), int)


class TestCropToNonzero(unittest.TestCase):
    def test_same_as_before(self):
        rng = np.random.default_rng(0)
        for data in _random_images(rng, 120):
            for view in _views(data):
                for with_seg in (False, True):
                    seg = rng.integers(0, 3, size=(1, *view.shape[1:])).astype(np.int8) if with_seg else None
                    with self.subTest(shape=view.shape, strides=view.strides, with_seg=with_seg):
                        d_new, s_new, b_new = crop_to_nonzero(view, None if seg is None else seg.copy())
                        d_old, s_old, b_old = _reference_crop_to_nonzero(view, None if seg is None else seg.copy())
                        self.assertEqual(b_new, b_old)
                        self.assertTrue(np.array_equal(d_new, d_old))
                        self.assertEqual(s_new.dtype, s_old.dtype)
                        self.assertTrue(np.array_equal(s_new, s_old))


if __name__ == '__main__':
    unittest.main()
