import unittest
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
from skimage.transform import resize

from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg


def _legacy_out_of_plane(reshaped_here: np.ndarray, new_shape, order_z: int) -> np.ndarray:
    """
    The out-of-plane step exactly as resample_data_or_seg used to do it: build a full (3, *new_shape)
    coordinate map and push it through map_coordinates. resample_data_or_seg now takes a 1d gather
    instead whenever order_z == 0, and this is the reference that has to reproduce bit for bit.
    """
    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
    orig_rows, orig_cols, orig_dim = reshaped_here.shape

    row_scale = float(orig_rows) / rows
    col_scale = float(orig_cols) / cols
    dim_scale = float(orig_dim) / dim

    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
    map_rows = row_scale * (map_rows + 0.5) - 0.5
    map_cols = col_scale * (map_cols + 0.5) - 0.5
    map_dims = dim_scale * (map_dims + 0.5) - 0.5

    coord_map = np.array([map_rows, map_cols, map_dims])
    return map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')


def _reference_resample(data, new_shape, is_seg=False, axis=None, order=3, do_separate_z=False,
                        order_z=0, internal_dtype=np.float64, legacy_out_of_plane=True):
    """
    resample_data_or_seg with the two things this module pins down made explicit. With the defaults
    (float64 working precision, full coordinate map) this is the implementation as it stood before
    either optimization, so a test against it covers both at once.
    """
    resize_fn, kwargs = (resize_segmentation, OrderedDict()) if is_seg \
        else (resize, {'mode': 'edge', 'anti_aliasing': False})
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=data.dtype)
    if not np.any(shape != new_shape):
        return data
    data = data.astype(internal_dtype, copy=False)

    if not do_separate_z:
        for c in range(data.shape[0]):
            reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        return reshaped_final

    assert axis is not None
    new_shape_2d = {0: new_shape[1:], 1: new_shape[[0, 2]], 2: new_shape[:-1]}[axis]
    for c in range(data.shape[0]):
        tmp = deepcopy(new_shape)
        tmp[axis] = shape[axis]
        reshaped_here = np.zeros(tmp, dtype=internal_dtype)
        for slice_id in range(shape[axis]):
            if axis == 0:
                reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
            elif axis == 1:
                reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
            else:
                reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
        if shape[axis] == new_shape[axis]:
            reshaped_final[c] = reshaped_here
        elif legacy_out_of_plane:
            reshaped_final[c] = _legacy_out_of_plane(reshaped_here, new_shape, order_z)
        else:
            scale = float(reshaped_here.shape[axis]) / new_shape[axis]
            coords = scale * (np.arange(new_shape[axis]) + 0.5) - 0.5
            indices = np.clip(np.floor(coords + 0.5).astype(np.intp), 0, reshaped_here.shape[axis] - 1)
            reshaped_final[c] = reshaped_here.take(indices, axis=axis)
    return reshaped_final


class TestOutOfPlaneGatherMatchesCoordinateMap(unittest.TestCase):
    """
    The two in-plane axes are already at their target size when the out-of-plane step runs, so their
    coordinate maps are exact identities and only `axis` is actually resampled. These tests pin that
    down: the cheap gather must agree with the coordinate map bit for bit, not approximately.
    """

    @staticmethod
    def _gather(reshaped_here, new_shape, axis):
        scale = float(reshaped_here.shape[axis]) / new_shape[axis]
        coords = scale * (np.arange(new_shape[axis]) + 0.5) - 0.5
        indices = np.clip(np.floor(coords + 0.5).astype(np.intp), 0, reshaped_here.shape[axis] - 1)
        return reshaped_here.take(indices, axis=axis)

    def test_gather_matches_map_coordinates_directly(self):
        """The isolated 1d gather against the isolated legacy coordinate map, nothing else around it."""
        rng = np.random.RandomState(1234)
        for _ in range(300):
            axis = rng.randint(0, 3)
            shape = [rng.randint(1, 12) for _ in range(3)]
            new_shape = list(shape)
            new_shape[axis] = rng.randint(1, 25)
            reshaped_here = rng.rand(*shape)

            gathered = self._gather(reshaped_here, new_shape, axis)
            legacy = _legacy_out_of_plane(reshaped_here, new_shape, order_z=0)
            self.assertEqual(gathered.shape, legacy.shape)
            self.assertTrue(np.array_equal(gathered, legacy),
                            f'mismatch for axis={axis} {shape} -> {new_shape}')

    def test_upsampling_and_downsampling_along_every_axis(self):
        """Both directions matter: coordinates land outside [0, n-1] when upsampling and must clamp."""
        for axis in range(3):
            for orig_len, new_len in ((7, 23), (23, 7), (1, 9), (9, 1), (16, 64), (100, 3)):
                shape, new_shape = [5, 6, 7], [5, 6, 7]
                shape[axis], new_shape[axis] = orig_len, new_len
                reshaped_here = np.random.RandomState(orig_len * 100 + new_len).rand(*shape)

                self.assertTrue(np.array_equal(self._gather(reshaped_here, new_shape, axis),
                                               _legacy_out_of_plane(reshaped_here, new_shape, 0)),
                                f'mismatch for axis={axis} {orig_len} -> {new_len}')


class TestResampleDataOrSegSeparateZ(unittest.TestCase):
    """End to end through resample_data_or_seg against the pre-optimization implementation."""

    def test_matches_legacy_on_random_configurations(self):
        """Covers the gather and the float32 working precision together: both must be bit-identical."""
        rng = np.random.RandomState(0)
        for _ in range(200):
            axis = rng.randint(0, 3)
            shape = [rng.randint(3, 12) for _ in range(3)]
            new_shape = list(shape)
            new_shape[axis] = rng.randint(3, 25)
            if rng.rand() < 0.5:  # the in-plane axes may be resampled too
                for a in range(3):
                    if a != axis:
                        new_shape[a] = rng.randint(3, 25)
            is_seg = bool(rng.rand() < 0.5)
            order = int(rng.choice([0, 1, 3]))
            n_channels = rng.randint(1, 3)
            data = (rng.randint(0, 4, (n_channels, *shape)).astype(np.float32) if is_seg
                    else rng.rand(n_channels, *shape).astype(np.float32))

            got = resample_data_or_seg(data.copy(), new_shape, is_seg=is_seg, axis=axis, order=order,
                                       do_separate_z=True, order_z=0)
            expected = _reference_resample(data.copy(), new_shape, is_seg=is_seg, axis=axis, order=order,
                                           do_separate_z=True, order_z=0)
            self.assertEqual(got.shape, expected.shape)
            self.assertTrue(np.array_equal(got, expected),
                            f'mismatch for axis={axis} {shape} -> {new_shape}, is_seg={is_seg}, order={order}')

    def test_order_z_other_than_zero_still_uses_the_coordinate_map(self):
        """order_z != 0 is untouched by the gather and must keep interpolating rather than gathering."""
        rng = np.random.RandomState(7)
        for is_seg in (False, True):
            data = (rng.randint(0, 3, (1, 6, 8, 8)).astype(np.float32) if is_seg
                    else rng.rand(1, 6, 8, 8).astype(np.float32))
            new_shape = [17, 8, 8]
            got = resample_data_or_seg(data.copy(), new_shape, is_seg=is_seg, axis=0, order=1,
                                       do_separate_z=True, order_z=1)
            self.assertEqual(tuple(got.shape), (1, *new_shape))
            if not is_seg:
                expected = _reference_resample(data.copy(), new_shape, is_seg=False, axis=0, order=1,
                                               do_separate_z=True, order_z=1)
                self.assertTrue(np.array_equal(got, expected))

    def test_shape_is_exact_for_awkward_ratios(self):
        """The gather's length comes from new_shape[axis] itself, so no ratio can round it off by one."""
        for orig_len, new_len in ((3, 7), (7, 3), (97, 991), (991, 97), (1, 1000), (1000, 1)):
            data = np.random.RandomState(orig_len).rand(1, orig_len, 4, 4).astype(np.float32)
            out = resample_data_or_seg(data, [new_len, 4, 4], is_seg=False, axis=0, order=1,
                                       do_separate_z=True, order_z=0)
            self.assertEqual(out.shape, (1, new_len, 4, 4))


class TestFloat32WorkingPrecision(unittest.TestCase):
    """
    resample_data_or_seg works in float32 rather than upcasting to float64. That is not an accuracy
    tradeoff: scipy.ndimage accumulates in double internally whatever the array dtype is, and for
    order > 1 it forces a float64 spline prefilter, so with float32 input the results are identical
    rather than merely close. These tests assert exact equality, so they will fail loudly if a future
    scipy/skimage stops doing that - at which point this needs revisiting, not a loosened tolerance.
    """

    def _assert_identical_to_float64(self, data, new_shape, **kwargs):
        got = resample_data_or_seg(data.copy(), new_shape, **kwargs)
        expected = _reference_resample(data.copy(), new_shape, internal_dtype=np.float64,
                                       legacy_out_of_plane=False, **kwargs)
        self.assertEqual(got.dtype, expected.dtype)
        self.assertTrue(np.array_equal(got, expected),
                        f'float32 differs from float64 for {kwargs}, '
                        f'max|diff| = {np.abs(got.astype(np.float64) - expected.astype(np.float64)).max():.3e}')

    def test_full_3d_path_every_order(self):
        rng = np.random.RandomState(11)
        for order in (0, 1, 3):
            for is_seg in (False, True):
                data = (rng.randint(0, 4, (1, 9, 12, 12)).astype(np.float32) if is_seg
                        else rng.randn(1, 9, 12, 12).astype(np.float32))
                self._assert_identical_to_float64(data, [21, 16, 16], is_seg=is_seg, order=order,
                                                  do_separate_z=False)

    def test_separate_z_path_every_order(self):
        rng = np.random.RandomState(12)
        for order in (0, 1, 3):
            for is_seg in (False, True):
                data = (rng.randint(0, 4, (1, 9, 12, 12)).astype(np.float32) if is_seg
                        else rng.randn(1, 9, 12, 12).astype(np.float32))
                self._assert_identical_to_float64(data, [21, 16, 16], is_seg=is_seg, axis=0, order=order,
                                                  do_separate_z=True, order_z=0)

    def test_long_axis_where_an_iir_spline_filter_would_accumulate_error(self):
        """order 3 prefiltering is recursive; a single precision filter would drift over a long axis."""
        data = np.random.RandomState(13).randn(1, 4, 4, 512).astype(np.float32)
        self._assert_identical_to_float64(data, [4, 4, 1024], is_seg=False, order=3, do_separate_z=False)

    def test_raw_hu_dynamic_range(self):
        """Unnormalized CT is the widest dynamic range this ever sees; float32 ulp is ~3e-4 there."""
        data = (np.random.RandomState(14).rand(1, 8, 16, 16) * 4000 - 1024).astype(np.float32)
        self._assert_identical_to_float64(data, [20, 20, 20], is_seg=False, order=3, do_separate_z=False)

    def test_segmentation_labels_are_unchanged(self):
        """The one that would actually hurt: a label flipping because of a rounding difference."""
        rng = np.random.RandomState(15)
        seg = rng.randint(0, 5, (1, 12, 24, 24)).astype(np.float32)
        for do_separate_z in (False, True):
            for order in (0, 1):
                got = resample_data_or_seg(seg.copy(), [30, 32, 32], is_seg=True, axis=0, order=order,
                                           do_separate_z=do_separate_z, order_z=0)
                expected = _reference_resample(seg.copy(), [30, 32, 32], is_seg=True, axis=0, order=order,
                                               do_separate_z=do_separate_z, order_z=0,
                                               internal_dtype=np.float64, legacy_out_of_plane=False)
                self.assertEqual(int((got != expected).sum()), 0,
                                 f'labels changed for separate_z={do_separate_z} order={order}')
                self.assertTrue(set(np.unique(got)).issubset(set(np.unique(seg))),
                                'resampling invented a label that was not in the input')

    def test_output_dtype_follows_input_dtype(self):
        for dtype in (np.float32, np.float64):
            data = np.random.RandomState(16).rand(1, 5, 6, 6).astype(dtype)
            out = resample_data_or_seg(data, [9, 8, 8], is_seg=False, order=1, do_separate_z=False)
            self.assertEqual(out.dtype, dtype)

    def test_float64_input_is_not_silently_downcast(self):
        """
        float64 callers must keep their precision. Values that differ only below the float32 ulp make
        this observable: downcasting first would collapse them onto each other.
        """
        rng = np.random.RandomState(17)
        data = (1.0 + rng.rand(1, 5, 6, 6) * 1e-12).astype(np.float64)
        self.assertEqual(len(np.unique(data.astype(np.float32))), 1,
                         'test is only meaningful if float32 cannot represent these differences')

        out = resample_data_or_seg(data.copy(), [11, 6, 6], is_seg=False, axis=0, order=0,
                                   do_separate_z=True, order_z=0)
        self.assertEqual(out.dtype, np.float64)
        self.assertGreater(len(np.unique(out)), 1, 'float64 input was downcast to float32 internally')

    def test_integer_input_is_converted_and_matches(self):
        """Nothing in nnU-Net feeds ints here today, but the cast has to keep working if something does."""
        seg = np.random.RandomState(18).randint(0, 4, (1, 8, 10, 10)).astype(np.int16)
        out = resample_data_or_seg(seg.copy(), [16, 12, 12], is_seg=True, axis=0, order=1,
                                   do_separate_z=True, order_z=0)
        expected = _reference_resample(seg.copy().astype(np.float32), [16, 12, 12], is_seg=True, axis=0,
                                       order=1, do_separate_z=True, order_z=0,
                                       internal_dtype=np.float64, legacy_out_of_plane=False)
        self.assertEqual(out.dtype, np.int16)
        self.assertTrue(np.array_equal(out, expected.astype(np.int16)))


if __name__ == '__main__':
    unittest.main()
