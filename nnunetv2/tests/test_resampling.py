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
            # segmentations are covered by TestSegResamplingDefault.test_out_of_plane_order_z_*

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


class TestSegResamplingTorch(unittest.TestCase):
    """
    resample_torch_simple has two seg implementations - the default one, which builds an
    (n_labels, c, *new_shape) score buffer, and the memory efficient one, which keeps a running
    argmax. They used to disagree: the memory efficient branch assigned with `interp > 0.5`, so a
    voxel no label claimed kept the zero `result` was initialized with, which is not necessarily a
    label that occurs in the input at all. They must now agree exactly, which also pins the float16
    score representation - a float32 running score resolves near ties the default branch collapses.
    """

    @staticmethod
    def _blobs(rng, shape, n_labels):
        v = sum(np.roll(rng.rand(*shape), rng.randint(0, 5), axis=rng.randint(0, len(shape)))
                for _ in range(3))
        q = np.quantile(v, np.linspace(0, 1, n_labels + 1)[1:-1]) if n_labels > 1 else []
        return np.digitize(v, q).astype(np.float32)[None]

    def test_both_branches_agree_exactly(self):
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        rng = np.random.RandomState(1234)
        for _ in range(60):
            ndim = int(rng.choice([2, 3]))
            shape = tuple(rng.randint(6, 24) for _ in range(ndim))
            data = self._blobs(rng, shape, rng.randint(1, 7))
            if rng.rand() < 0.2:  # label values need not start at zero
                data = data + rng.randint(1, 4)
            new_shape = [max(int(round(s / f)), 1)
                         for s, f in zip(shape, rng.choice([1., 1.5, 2., 2.5, 3., 4., .5, .75], ndim))]
            for tiebreak in ('nearest', 'lowest', 'highest'):
                default = np.asarray(resample_torch_simple(data.copy(), new_shape, is_seg=True,
                                                           memefficient_seg_resampling=False,
                                                           seg_tiebreak=tiebreak))
                memeff = np.asarray(resample_torch_simple(data.copy(), new_shape, is_seg=True,
                                                          memefficient_seg_resampling=True,
                                                          seg_tiebreak=tiebreak))
                self.assertTrue(np.array_equal(default, memeff),
                                f'branches disagree for {shape} -> {new_shape}, tiebreak={tiebreak}')

    def test_no_label_is_invented(self):
        """The regression that motivated the rewrite: 11111222 downsampled 2x used to yield 1102."""
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        for pattern in ('11111222', '22233333', '11122333'):
            data = np.array([int(c) for c in pattern], dtype=np.float32)[None, :, None].repeat(2, axis=2)
            for memefficient in (False, True):
                for tiebreak in ('nearest', 'lowest', 'highest'):
                    out = np.asarray(resample_torch_simple(data.copy(), (4, 2), is_seg=True,
                                                           memefficient_seg_resampling=memefficient,
                                                           seg_tiebreak=tiebreak))
                    self.assertTrue(set(np.unique(out).astype(int)) <= set(int(c) for c in pattern),
                                    f'{pattern} produced labels that are not in the input: '
                                    f'{np.unique(out)} (memefficient={memefficient}, {tiebreak})')

    def test_nearest_tiebreak_only_picks_among_the_tied_labels(self):
        """
        Where three or more labels meet, the nearest neighbour can be a label that lost: halving this 2x2x2
        block scores 1 and 2 at 0.375 each and 3 at 0.25, and the nearest neighbour of the centre is a 3.
        """
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        block = np.array([3, 1, 1, 1, 2, 2, 2, 3], dtype=np.float32).reshape(2, 2, 2)
        data = np.tile(block, (8, 8, 8))[None]
        for memefficient in (False, True):
            for tiebreak, expected in (('nearest', 1), ('lowest', 1), ('highest', 2)):
                out = np.asarray(resample_torch_simple(data.copy(), (8, 8, 8), is_seg=True,
                                                       memefficient_seg_resampling=memefficient,
                                                       seg_tiebreak=tiebreak))
                self.assertTrue((out == expected).all(),
                                f'{np.unique(out)} (memefficient={memefficient}, {tiebreak})')
        out = resample_data_or_seg(data.copy(), (8, 8, 8), is_seg=True, order=1, seg_tiebreak='nearest')
        self.assertTrue((out == 1).all(), f'default path: {np.unique(out)}')

    def test_winner_has_the_top_score(self):
        """whatever settles a tie, the label a voxel gets must attain the maximum (float16) score there"""
        import torch
        from torch.nn import functional as F
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        rng = np.random.RandomState(99)
        for i in range(24):
            shape = tuple(int(rng.randint(4, 10)) * 2 for _ in range(3))
            n = int(rng.randint(3, 6))
            data = (rng.randint(0, n, shape) if i % 2 else self._blobs(rng, shape, n)[0]).astype(np.float32)[None]
            new_shape = [s // 2 for s in shape] if i % 3 else [max(int(round(s / 1.5)), 1) for s in shape]
            labels = np.unique(data)
            scores = torch.stack([F.interpolate(torch.from_numpy((data == u).astype(np.float32) * 1000)[None],
                                                new_shape, mode='trilinear')[0].half() for u in labels])
            top = scores.max(0).values
            for memefficient in (False, True):
                for tiebreak in ('nearest', 'lowest', 'highest'):
                    out = torch.as_tensor(np.asarray(resample_torch_simple(
                        data.copy(), new_shape, is_seg=True, memefficient_seg_resampling=memefficient,
                        seg_tiebreak=tiebreak))).float()
                    own = scores.gather(0, torch.searchsorted(torch.from_numpy(labels), out)[None])[0]
                    self.assertTrue(torch.equal(own, top), f'{int((own != top).sum())} voxels '
                                                           f'(memefficient={memefficient}, {tiebreak}, i={i})')

    def test_tiebreak_directions(self):
        """
        An even integer factor makes every boundary voxel an exact 50/50 tie. 'lowest' hands all of
        them to the smallest label, which erodes foreground; 'nearest' takes the nearest neighbour,
        which is what the separate-z path does and is symmetric in the labels.
        """
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple

        def run(pattern, **kwargs):
            data = np.array([int(c) for c in pattern], dtype=np.float32)[None, :, None].repeat(2, axis=2)
            out = np.asarray(resample_torch_simple(data, (len(pattern) // 2, 2), is_seg=True, **kwargs))
            return ''.join(str(int(round(v))) for v in out[0, :, 0])

        for memefficient in (False, True):
            kw = {'memefficient_seg_resampling': memefficient}
            self.assertEqual(run('00011111', seg_tiebreak='lowest', **kw), '0011')
            self.assertEqual(run('00011111', seg_tiebreak='nearest', **kw), '0111')
            self.assertEqual(run('11111222', seg_tiebreak='lowest', **kw), '1112')
            self.assertEqual(run('11111222', seg_tiebreak='nearest', **kw), '1122')
            self.assertEqual(run('00011111', seg_tiebreak='highest', **kw), '0111')
            self.assertEqual(run('22233333', seg_tiebreak='highest', **kw), '2333')
            self.assertEqual(run('22233333', seg_tiebreak='lowest', **kw), '2233')
            # no tie here, so both must agree
            self.assertEqual(run('00001111', seg_tiebreak='lowest', **kw),
                             run('00001111', seg_tiebreak='nearest', **kw))

    def test_roundtrip_is_exact_when_the_label_map_is_block_aligned(self):
        """00001111 -> 0011 -> 00001111. Down and up share the same block grid, so a segmentation
        that is constant on those blocks has to survive the roundtrip untouched."""
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        for tiebreak in ('nearest', 'lowest'):
            data = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)[None, :, None].repeat(2, axis=2)
            down = np.asarray(resample_torch_simple(data, (4, 2), is_seg=True, seg_tiebreak=tiebreak))
            back = np.asarray(resample_torch_simple(down.astype(np.float32), (8, 2), is_seg=True,
                                                    seg_tiebreak=tiebreak))
            self.assertTrue(np.array_equal(back.astype(np.float32), data), f'tiebreak={tiebreak}')

    def test_mode_is_forwarded_when_not_separating_z(self):
        """resample_torch_fornnunet used to call resample_torch_simple positionally and stop one
        argument short, so a caller supplied mode was silently replaced by the default."""
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
        data = (np.arange(8, dtype=np.float32)[None, :, None, None] *
                np.ones((1, 8, 2, 2), dtype=np.float32))
        kwargs = dict(current_spacing=(1., 1., 1.), new_spacing=(1., 1., 1.), is_seg=False,
                      force_separate_z=False)
        got = {m: resample_torch_fornnunet(data, [4, 2, 2], mode=m, **kwargs)[0, :, 0, 0]
               for m in ('linear', 'nearest-exact')}
        self.assertTrue(np.allclose(got['linear'], [0.5, 2.5, 4.5, 6.5]))
        self.assertTrue(np.allclose(got['nearest-exact'], [1., 3., 5., 7.]))


def _blob_seg(rng, shape, n_labels, offset=0):
    """A label map with spatially coherent regions, like a real segmentation."""
    v = sum(np.roll(rng.rand(*shape), rng.randint(0, 5), axis=rng.randint(0, len(shape)))
            for _ in range(3))
    q = np.quantile(v, np.linspace(0, 1, n_labels + 1)[1:-1]) if n_labels > 1 else []
    return (np.digitize(v, q).astype(np.float32) + offset)[None]


class TestUniqueLabels(unittest.TestCase):
    """
    unique_labels replaced torch.unique in the seg paths because pd.unique hashes where torch sorts.
    The running argmax walks the labels in ascending order and resolves a strict `>` chain to the
    first one that attains the maximum, so both the values AND their order are load bearing.
    """

    def test_matches_torch_unique(self):
        import torch
        from nnunetv2.preprocessing.resampling.resample_torch import unique_labels
        rng = np.random.RandomState(11)
        for dtype in (torch.float32, torch.int8, torch.int16, torch.float64):
            for labels in ([0], [0, 1], [-1, 0, 1, 2], [0, 5, 200], list(range(60)), [3]):
                a = np.array(labels, dtype=np.int64)[rng.randint(0, len(labels), 4000)]
                if dtype == torch.int8 and (a.max() > 127 or a.min() < -128):
                    continue
                t = torch.from_numpy(a).to(dtype)
                got, want = unique_labels(t), torch.unique(t)
                self.assertEqual(got.dtype, want.dtype, f'{dtype} {labels}')
                self.assertTrue(torch.equal(got, want), f'{dtype} {labels}: {got} vs {want}')

    def test_handles_non_contiguous_input(self):
        import torch
        from nnunetv2.preprocessing.resampling.resample_torch import unique_labels
        t = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) % 4
        view = t.permute(2, 0, 1)[::2]
        self.assertTrue(torch.equal(unique_labels(view), torch.unique(view)))


class TestInterpolateInto(unittest.TestCase):
    """
    _interpolate_into exists only because F.interpolate has no out=. It must dispatch to the same
    kernel, or the scores change and with them the argmax.
    """

    def test_bit_identical_to_functional_interpolate(self):
        import torch
        from torch.nn import functional as F
        from nnunetv2.preprocessing.resampling.resample_torch import _interpolate_into
        rng = np.random.RandomState(12)
        for ndim, modes in ((3, ('trilinear', 'nearest', 'nearest-exact')),
                            (2, ('bilinear', 'nearest', 'nearest-exact'))):
            for _ in range(6):
                shape = tuple(int(rng.randint(4, 12)) for _ in range(ndim))
                new_shape = [max(int(round(s * f)), 1) for s, f in
                             zip(shape, rng.choice([0.5, 0.75, 1.5, 2., 3.], ndim))]
                x = torch.from_numpy(rng.rand(1, 2, *shape).astype(np.float32))
                for mode in modes:
                    kw = {'align_corners': False} if mode in ('trilinear', 'bilinear') else {}
                    want = F.interpolate(x, new_shape, mode=mode, **kw)
                    got = _interpolate_into(x, new_shape, mode,
                                            torch.empty_like(want).fill_(float('nan')))
                    self.assertTrue(torch.equal(got, want), f'{mode} {shape}->{new_shape}')

    def test_modes_without_an_out_kernel_fall_back_to_functional_interpolate(self):
        import torch
        from torch.nn import functional as F
        from nnunetv2.preprocessing.resampling.resample_torch import _interpolate_into
        rng = np.random.RandomState(16)
        for mode, ndim in (('bicubic', 2), ('area', 2), ('area', 3)):
            x = torch.from_numpy(rng.rand(1, 2, *([7] * ndim)).astype(np.float32))
            want = F.interpolate(x, [5] * ndim, mode=mode)
            got = _interpolate_into(x, [5] * ndim, mode, torch.empty_like(want).fill_(float('nan')))
            self.assertTrue(torch.equal(got, want), mode)


class TestSegRunningArgmaxInternals(unittest.TestCase):
    """
    The memory efficient branch was rewritten to hoist every per-label temporary out of the loop, to
    restrict each label to the channels it occupies, and to short circuit a single-label volume. None
    of that may change a voxel.
    """

    def test_channel_restriction_matches_per_channel_resampling(self):
        """
        Dim 0 is never interpolated, so resampling a stack must equal resampling each channel on its
        own. This is what makes restricting a label to its channel range sound - and it is the hot
        path of separate-z, where the channel dim carries the z slices.
        """
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        rng = np.random.RandomState(13)
        for _ in range(8):
            n_ch = int(rng.randint(3, 9))
            shape = (int(rng.randint(6, 14)), int(rng.randint(6, 14)))
            # each channel carries a different subset of the labels, so the channel ranges are
            # genuinely disjoint rather than all spanning the stack
            stack = np.concatenate([_blob_seg(rng, shape, int(rng.randint(1, 4)),
                                              offset=int(rng.randint(0, 3)))
                                    for _ in range(n_ch)], axis=0)
            new_shape = [max(int(round(s * f)), 1) for s, f in
                         zip(shape, rng.choice([0.5, 1.5, 2., 3.], 2))]
            for tiebreak in ('nearest', 'lowest'):
                together = np.asarray(resample_torch_simple(stack.copy(), new_shape, is_seg=True,
                                                            seg_tiebreak=tiebreak))
                apart = np.concatenate([
                    np.asarray(resample_torch_simple(stack[i:i + 1].copy(), new_shape, is_seg=True,
                                                     seg_tiebreak=tiebreak))
                    for i in range(n_ch)], axis=0)
                self.assertTrue(np.array_equal(together, apart),
                                f'channel stacking changed the result ({tiebreak})')

    def test_single_label_volume(self):
        """The short circuit must return that label everywhere, in the dtype the loop would pick."""
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        for value, expect_dtype in ((0, np.int8), (7, np.int8), (300, np.int16)):
            data = np.full((1, 6, 7, 8), float(value), dtype=np.float32)
            out = np.asarray(resample_torch_simple(data, [9, 5, 8], is_seg=True))
            self.assertEqual(out.dtype, expect_dtype)
            self.assertEqual(out.shape, (1, 9, 5, 8))
            self.assertTrue(np.all(out == value))

    def test_branches_agree_when_ties_are_everywhere(self):
        """
        An exact integer factor puts a tie on every boundary voxel, which is what exercises the tie
        bookkeeping and the scratch buffers that share the score buffer's storage. The two seg
        branches must still agree exactly.
        """
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        rng = np.random.RandomState(14)
        for n_labels in (2, 3, 7, 12):
            data = _blob_seg(rng, (8, 9, 10), n_labels)
            for factor in (2, 4):
                new_shape = [s * factor for s in data.shape[1:]]
                for tiebreak in ('nearest', 'lowest', 'highest'):
                    a = np.asarray(resample_torch_simple(data.copy(), new_shape, is_seg=True,
                                                         memefficient_seg_resampling=False,
                                                         seg_tiebreak=tiebreak))
                    b = np.asarray(resample_torch_simple(data.copy(), new_shape, is_seg=True,
                                                         memefficient_seg_resampling=True,
                                                         seg_tiebreak=tiebreak))
                    self.assertTrue(np.array_equal(a, b),
                                    f'{n_labels} labels, {factor}x, {tiebreak}')

    def test_unknown_tiebreak_is_rejected(self):
        """
        Up front, on every path - not only once some voxel happens to need a tie settled, which depends on
        the data and would let a typo through on some cases of a dataset and fail on others.
        """
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple, \
            resample_torch_fornnunet
        data = _blob_seg(np.random.RandomState(15), (6, 6, 6), 3)
        cases = [dict(memefficient_seg_resampling=m) for m in (True, False)] + [
            dict(mode='nearest'),  # the gather, which never looks at seg_tiebreak
            dict(mode='nearest-exact'),
        ]
        for kw in cases:
            with self.assertRaises(ValueError, msg=str(kw)):
                resample_torch_simple(data.copy(), [7, 7, 7], is_seg=True, seg_tiebreak='whatever', **kw)
        # a single-label volume, which is settled before any label is scored
        with self.assertRaises(ValueError):
            resample_torch_simple(np.full((1, 4, 4, 4), 3, np.float32), [7, 7, 7], is_seg=True,
                                  seg_tiebreak='whatever')
        with self.assertRaises(ValueError):
            resample_torch_fornnunet(data.copy(), [7, 7, 7], [1, 1, 1], [1, 1, 1], is_seg=True,
                                     seg_tiebreak='whatever')

    def test_highest_is_accepted_like_on_the_default_path(self):
        """A plans file that works with the default resampler must not crash the torch one."""
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
        data = _blob_seg(np.random.RandomState(17), (6, 8, 8), 3)
        for force_separate_z in (False, None):
            out = resample_torch_fornnunet(data.copy(), [12, 16, 16], [3, 1, 1], [1.5, .5, .5], is_seg=True,
                                           force_separate_z=force_separate_z, seg_tiebreak='highest')
            self.assertTrue(set(np.unique(out)) <= set(np.unique(data)))

    def test_modes_without_an_out_kernel(self):
        """
        The memory efficient branch is the default, and it used to raise for any mode _interpolate_into
        had no out= kernel for. Both branches must handle them, and agree.
        """
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        rng = np.random.RandomState(18)
        for mode, shape, new_shape in (('area', (12, 12, 12), [6, 6, 6]), ('area', (12, 14), [6, 7]),
                                       ('bicubic', (12, 14), [9, 21])):
            data = _blob_seg(rng, shape, 4)
            for tiebreak in ('nearest', 'lowest', 'highest'):
                a, b = (np.asarray(resample_torch_simple(data.copy(), new_shape, is_seg=True, mode=mode,
                                                         memefficient_seg_resampling=m, seg_tiebreak=tiebreak))
                        for m in (False, True))
                self.assertEqual(a.shape, (1, *new_shape))
                self.assertTrue(np.array_equal(a, b), f'{mode} {shape} {tiebreak}')


class TestBothResamplersShareInvariants(unittest.TestCase):
    """
    Properties that must hold for the scipy resampler and the torch resampler alike. They are
    different algorithms and do not agree voxel for voxel, but a segmentation resampler that breaks
    any of these is broken whichever one it is.
    """

    @staticmethod
    def _resamplers():
        from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet

        def scipy_fn(data, new_shape, sp, tsp, is_seg, seg_tiebreak='nearest'):
            return resample_data_or_seg_to_shape(data, new_shape, sp, tsp, is_seg=is_seg,
                                                 order=1 if is_seg else 3, order_z=0,
                                                 force_separate_z=None, seg_tiebreak=seg_tiebreak)

        def torch_fn(data, new_shape, sp, tsp, is_seg, seg_tiebreak='nearest'):
            return resample_torch_fornnunet(data, new_shape, sp, tsp, is_seg=is_seg,
                                            force_separate_z=None, seg_tiebreak=seg_tiebreak)

        return (('scipy', scipy_fn), ('torch', torch_fn))

    def test_no_label_is_invented(self):
        rng = np.random.RandomState(21)
        for name, fn in self._resamplers():
            for _ in range(6):
                shape = tuple(int(rng.randint(6, 14)) for _ in range(3))
                data = _blob_seg(rng, shape, int(rng.randint(2, 6)), offset=int(rng.choice([0, 1])))
                sp = np.array([float(rng.choice([1., 5.])), 1., 1.])
                tsp = sp / float(rng.choice([0.5, 1.5, 2.]))
                new_shape = [max(int(round(s * a / b)), 1) for s, a, b in zip(shape, sp, tsp)]
                out = np.asarray(fn(data.copy(), new_shape, sp, tsp, True))
                self.assertTrue(set(np.unique(out).astype(int)) <= set(np.unique(data).astype(int)),
                                f'{name} invented labels: {np.unique(out)} from {np.unique(data)}')

    def test_same_shape_is_a_no_op(self):
        rng = np.random.RandomState(22)
        data = _blob_seg(rng, (7, 8, 9), 4)
        sp = np.array([1., 1., 1.])
        for name, fn in self._resamplers():
            out = np.asarray(fn(data.copy(), list(data.shape[1:]), sp, sp, True))
            self.assertTrue(np.array_equal(out.astype(np.int64), data.astype(np.int64)),
                            f'{name} changed a segmentation it was asked not to resample')

    def test_block_aligned_upsample_roundtrip_is_exact(self):
        """An integer upsample followed by the matching downsample must return the original."""
        rng = np.random.RandomState(23)
        data = _blob_seg(rng, (6, 7, 8), 4)
        sp = np.array([1., 1., 1.])
        for name, fn in self._resamplers():
            for factor in (2, 3):
                up_shape = [s * factor for s in data.shape[1:]]
                up = fn(data.copy(), up_shape, sp, sp / factor, True)
                back = np.asarray(fn(np.asarray(up).astype(np.float32), list(data.shape[1:]),
                                     sp / factor, sp, True))
                self.assertTrue(np.array_equal(back.astype(np.int64), data.astype(np.int64)),
                                f'{name} roundtrip at {factor}x is not exact')

    def test_tiebreak_lowest_never_beats_nearest_on_foreground(self):
        """
        At an even integer factor every boundary voxel is an exact tie. 'lowest' hands them all to
        the smallest label, which for a normal segmentation is background, so it can only ever have
        less foreground than 'nearest' - never more. Both resamplers must show that direction.
        """
        rng = np.random.RandomState(24)
        sp = np.array([1., 1., 1.])
        for name, fn in self._resamplers():
            total_low = total_near = 0
            for _ in range(4):
                data = _blob_seg(rng, (8, 8, 8), 3)
                new_shape = [s * 2 for s in data.shape[1:]]
                low = np.asarray(fn(data.copy(), new_shape, sp, sp / 2, True, seg_tiebreak='lowest'))
                near = np.asarray(fn(data.copy(), new_shape, sp, sp / 2, True, seg_tiebreak='nearest'))
                total_low += int((low > 0).sum())
                total_near += int((near > 0).sum())
            self.assertLessEqual(total_low, total_near,
                                 f'{name}: lowest produced more foreground than nearest')

    def test_image_path_shape_and_dtype(self):
        rng = np.random.RandomState(25)
        data = rng.rand(2, 7, 8, 9).astype(np.float32)
        sp = np.array([1., 1., 1.])
        for name, fn in self._resamplers():
            out = np.asarray(fn(data.copy(), [14, 8, 5], sp, np.array([0.5, 1., 1.8]), False))
            self.assertEqual(out.shape, (2, 14, 8, 5), f'{name}')
            self.assertEqual(out.dtype, np.float32, f'{name}')
            self.assertTrue(np.all(np.isfinite(out)), f'{name} produced non-finite values')

class TestSegOutOfPlaneNearest(unittest.TestCase):
    """
    resample_torch_fornnunet resamples the anisotropic axis with a nearest mode. That used to go
    through resample_torch_simple's seg path, which scores one full sized float32 interpolation per
    label and then argmaxes - to compute what a nearest interpolation gives directly. Nearest can
    only copy a value that is already in the input and leaves no voxel undecided, so the one-hot
    detour was pure cost: 16-76x this step on real data, over half of the whole separate-z call.
    The two must stay bit-identical, including the returned integer dtype.
    """

    @staticmethod
    def _seg(rng, shape, n_labels):
        v = sum(np.roll(rng.rand(*shape), rng.randint(0, 5), axis=rng.randint(0, len(shape)))
                for _ in range(3))
        q = np.quantile(v, np.linspace(0, 1, n_labels + 1)[1:-1]) if n_labels > 1 else []
        return np.digitize(v, q).astype(np.float32)[None]

    def test_matches_the_one_hot_path(self):
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
        rng = np.random.RandomState(9876)
        for _ in range(40):
            shape = tuple(rng.randint(5, 16) for _ in range(3))
            data = self._seg(rng, shape, rng.randint(1, 8))
            if rng.rand() < 0.3:  # label values need not start at zero
                data = data + rng.randint(1, 4)
            new_shape = [max(int(round(s * f)), 1)
                         for s, f in zip(shape, rng.choice([1., 1.5, 2., 2.5, 3., .5, .75], 3))]
            for mode in ('nearest', 'nearest-exact'):
                for tiebreak in ('nearest', 'lowest'):
                    one_hot = resample_torch_simple(data.copy(), new_shape, is_seg=True, mode=mode,
                                                    seg_tiebreak=tiebreak)
                    direct = resample_torch_simple(data.copy(), new_shape, is_seg=False, mode=mode)
                    self.assertTrue(np.array_equal(np.asarray(one_hot),
                                                   np.asarray(direct).astype(one_hot.dtype)),
                                    f'{shape} -> {new_shape}, mode={mode}, tiebreak={tiebreak}')

    def test_separate_z_return_dtype_is_still_integer(self):
        """The shortcut has to cast back: callers store the result as int8/int16."""
        import torch
        from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
        rng = np.random.RandomState(4321)
        data = self._seg(rng, (8, 20, 20), 5)
        for offset, expected in ((0, torch.int8), (200, torch.int16)):
            out = resample_torch_fornnunet(data + offset, [16, 15, 15], (5., 1., 1.), (2.5, 1., 1.),
                                           is_seg=True, force_separate_z=None)
            self.assertEqual(torch.from_numpy(out).dtype, expected)
            self.assertTrue(set(np.unique(out)) <= set(np.unique(data + offset)))


class TestSegResamplingDefault(unittest.TestCase):
    """
    The default (non-torch) resampling path reaches batchgenerators' resize_segmentation, which used to
    assign labels with `interpolated >= 0.5` into a zero-initialized result. seg_tiebreak now settles the
    voxels where two labels share the top interpolated indicator - which is every boundary voxel when an
    axis is resampled by an even integer factor - and the default 'nearest' matches resample_torch.
    """

    def test_tiebreak_reaches_the_default_resampling_path(self):
        seg = np.zeros((1, 8, 4, 4), dtype=np.uint8)
        seg[0, 5:] = 1  # the boundary lands exactly on an output centre when we halve the first axis
        got = {tb: resample_data_or_seg(seg.copy(), [4, 4, 4], is_seg=True, order=1, do_separate_z=False,
                                        seg_tiebreak=tb)[0, :, 0, 0]
               for tb in ('nearest', 'lowest', 'highest')}
        # 'lowest' hands the tied voxel to the background and erodes the structure, which is what the
        # tie-break exists to stop; 'nearest' keeps it
        self.assertTrue(np.array_equal(got['lowest'], [0, 0, 0, 1]))
        self.assertTrue(np.array_equal(got['nearest'], [0, 0, 1, 1]))
        self.assertTrue(np.array_equal(got['highest'], [0, 0, 1, 1]))

    def test_default_is_nearest(self):
        seg = np.zeros((1, 8, 4, 4), dtype=np.uint8)
        seg[0, 5:] = 1
        default = resample_data_or_seg(seg.copy(), [4, 4, 4], is_seg=True, order=1, do_separate_z=False)
        explicit = resample_data_or_seg(seg.copy(), [4, 4, 4], is_seg=True, order=1, do_separate_z=False,
                                        seg_tiebreak='nearest')
        self.assertTrue(np.array_equal(default, explicit))

    def test_no_label_is_invented(self):
        rng = np.random.RandomState(1234)
        for _ in range(20):
            seg = rng.randint(1, 5, (1, 12, 12, 12)).astype(np.uint8)  # labels 1..4, never 0
            for tiebreak in ('nearest', 'lowest', 'highest'):
                out = resample_data_or_seg(seg.copy(), [6, 6, 6], is_seg=True, order=1, do_separate_z=False,
                                           seg_tiebreak=tiebreak)
                self.assertTrue(np.isin(out, np.unique(seg)).all(),
                                f'{np.unique(out)} contains a label that is not in the input ({tiebreak})')

    def test_unknown_tiebreak_is_rejected(self):
        seg = np.zeros((1, 8, 4, 4), dtype=np.uint8)
        for order in (0, 1):  # resize_segmentation never looks at seg_tiebreak at order 0
            with self.assertRaises(ValueError):
                resample_data_or_seg(seg.copy(), [4, 4, 4], is_seg=True, order=order, seg_tiebreak='nearst')
        # images do not care
        resample_data_or_seg(seg.astype(np.float32), [4, 4, 4], is_seg=False, order=1, seg_tiebreak='nearst')

    def test_out_of_plane_order_z_matches_the_argmax(self):
        """
        order_z != 0 on the separate-z path resamples `axis` with the argmax too, not with the old per-label
        `> 0.5` threshold. Against an explicit argmax over the interpolated indicators, which is what that
        threshold was supposed to be.
        """
        rng = np.random.RandomState(19)
        for _ in range(20):
            axis = int(rng.randint(0, 3))
            shape = [int(rng.randint(4, 9)) for _ in range(3)]
            new_shape = list(shape)
            new_shape[axis] = int(rng.randint(3, 17))
            seg = rng.randint(1, 5, (1, *shape)).astype(np.float32)
            for tiebreak in ('lowest', 'highest'):
                got = resample_data_or_seg(seg.copy(), new_shape, is_seg=True, axis=axis, order=1,
                                           do_separate_z=True, order_z=1, seg_tiebreak=tiebreak)[0]
                labels = np.unique(seg)
                scores = np.stack([_legacy_out_of_plane((seg[0] == u).astype(np.float64), new_shape, 1)
                                   for u in labels])
                winner = scores.argmax(0) if tiebreak == 'lowest' else len(labels) - 1 - scores[::-1].argmax(0)
                self.assertTrue(np.array_equal(got, labels[winner]), f'{shape} -> {new_shape}, axis {axis}')

    def test_out_of_plane_order_z_invents_no_label_and_takes_the_tiebreak(self):
        # three labels meeting along axis 0: 1 | 2 | 3, upsampled so that a sample point falls where no
        # single indicator reaches 0.5 - which the old threshold left at 0
        seg = np.zeros((1, 3, 2, 2), dtype=np.float32)
        seg[0, 0], seg[0, 1], seg[0, 2] = 1, 2, 3
        for tiebreak in ('nearest', 'lowest', 'highest'):
            out = resample_data_or_seg(seg.copy(), [7, 2, 2], is_seg=True, axis=0, order=1, do_separate_z=True,
                                       order_z=1, seg_tiebreak=tiebreak)
            self.assertTrue(set(np.unique(out)) <= {1, 2, 3}, f'{np.unique(out)} ({tiebreak})')
        # 1 1 1 2 2 downsampled 2x along axis 0 has an exact tie at the centre of voxels 2 and 3
        seg = np.array([1, 1, 1, 2, 2, 2, 2, 2], dtype=np.float32)[None, :, None, None].repeat(2, 2).repeat(2, 3)
        got = {tb: resample_data_or_seg(seg.copy(), [4, 2, 2], is_seg=True, axis=0, order=1, do_separate_z=True,
                                        order_z=1, seg_tiebreak=tb)[0, :, 0, 0].tolist()
               for tb in ('nearest', 'lowest', 'highest')}
        self.assertEqual(got['lowest'], [1, 1, 2, 2])
        self.assertEqual(got['highest'], [1, 2, 2, 2])
        self.assertIn(got['nearest'], (got['lowest'], got['highest']))

    def test_separate_z_path_also_takes_the_tiebreak(self):
        seg = np.zeros((1, 8, 8, 4), dtype=np.uint8)
        seg[0, :, 5:] = 1
        got = {tb: resample_data_or_seg(seg.copy(), [8, 4, 4], is_seg=True, axis=0, order=1,
                                        do_separate_z=True, order_z=0, seg_tiebreak=tb)[0, 0, :, 0]
               for tb in ('nearest', 'lowest')}
        self.assertFalse(np.array_equal(got['nearest'], got['lowest']),
                         'seg_tiebreak is not reaching the in-plane resize on the separate-z path')


if __name__ == '__main__':
    unittest.main()
