import unittest
from types import SimpleNamespace

import numpy as np

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class _OneVoxelLocations:
    """Foreground locations of a single case whose only foreground voxel is self.voxel."""
    class_keys = [1]

    def __init__(self, annotated_classes_key):
        self.annotated_classes_key = annotated_classes_key
        self.voxel = None

    def eligible_classes(self, identifier):
        return [1]

    def count(self, identifier, class_key):
        return 1

    def sample(self, identifier, class_key):
        assert class_key in (1, self.annotated_classes_key)
        return self.voxel


_NOT_GIVEN = object()


class _BboxTestCase(unittest.TestCase):
    final_patch_size = (16, 16, 16)
    # the initial patch is larger so that rotations do not pull in padding; the final patch is its center. Even
    # margins, so that the center is exact
    patch_size = (24, 20, 28)

    def _loader(self, labels, x=_NOT_GIVEN, patch_size=None, final_patch_size=None, p=_NOT_GIVEN):
        lm = LabelManager(labels, None)
        locations = _OneVoxelLocations(lm.annotated_classes_key)
        data = SimpleNamespace(identifiers=['case'], foreground_locations=locations)
        kwargs = {} if x is _NOT_GIVEN else {'min_patch_range_div': x}
        if p is not _NOT_GIVEN:
            kwargs['min_patch_range_p'] = p
        dl = nnUNetDataLoader(data, 1, patch_size or self.patch_size, final_patch_size or self.final_patch_size, lm,
                              1.0, **kwargs)
        return dl, locations

    def _final_patch_starts(self, dl, case_shape, force_fg):
        bbox_lbs, bbox_ubs = dl.get_bbox('case', np.array(case_shape), force_fg)
        for d in range(len(case_shape)):
            self.assertEqual(bbox_ubs[d] - bbox_lbs[d], dl.patch_size[d])
        return [bbox_lbs[d] + (dl.patch_size[d] - dl.final_patch_size[d]) // 2 for d in range(len(case_shape))]


class TestMinimumPatchRange(_BboxTestCase):
    """
    Along every axis the final patch can move by at least final_patch_size // X (min_patch_range_div, default 2).
    Where the range of patch positions inside the case (case size - final patch size, or a single centered position
    for cases smaller than the patch) is shorter, it is widened to that, equally on both sides. Longer ranges stay.
    By default this is applied to half the samples it would change (min_patch_range_p, TestMinimumRangeProbability);
    the tests here apply it to all of them (p = 1).
    """
    # smaller than the patch, exactly the patch, 1 / 2 / 4 voxels of movement inside the case, plenty of movement
    case_shapes = ((10, 16, 17), (18, 20, 40))

    def _expected_range(self, s, f, x):
        """-> lowest and highest start of the final patch along an axis of case size s"""
        if s >= f:
            lo, hi = 0, s - f  # inside the case
        else:
            lo, hi = -((f - s + 1) // 2), -((f - s) // 2)  # centered (one voxel either way if odd)
        deficit = f // x - (hi - lo)
        if deficit > 0:
            lo, hi = lo - deficit // 2, hi + deficit - deficit // 2
        return lo, hi

    def test_random_patches(self):
        np.random.seed(1)
        for x in (2, 4, 8):
            dl, _ = self._loader({'background': 0, 'a': 1}, x, p=1.0)
            for case_shape in self.case_shapes:
                starts = np.array([self._final_patch_starts(dl, case_shape, False) for _ in range(2000)])
                for d in range(3):
                    expected = self._expected_range(case_shape[d], self.final_patch_size[d], x)
                    with self.subTest(x=x, case_size=case_shape[d]):
                        self.assertEqual((starts[:, d].min(), starts[:, d].max()), expected)
                        # the point of it: never less movement than final_patch_size // x
                        self.assertGreaterEqual(expected[1] - expected[0], self.final_patch_size[d] // x)

    def test_default_is_half_the_final_patch_in_half_the_samples(self):
        dl, _ = self._loader({'background': 0, 'a': 1})
        self.assertEqual(dl.min_patch_range_div, 2)
        self.assertEqual(dl.min_patch_range, [8, 8, 8])
        self.assertEqual(dl.min_patch_range_p, 0.5)
        np.random.seed(3)
        case_shape = (10, 16, 17)
        starts = np.array([self._final_patch_starts(dl, case_shape, False) for _ in range(2000)])
        for d in range(3):
            self.assertEqual((starts[:, d].min(), starts[:, d].max()),
                             self._expected_range(case_shape[d], self.final_patch_size[d], 2))

    def test_large_cases_unchanged(self):
        # 40 voxels: 24 positions of movement inside the case, more than any x asks for
        for x in (2, 4, 8):
            self.assertEqual(self._expected_range(40, 16, x), (0, 24))
        # 20 voxels: 4 positions, enough for x = 4 and 8, widened for x = 2
        self.assertEqual(self._expected_range(20, 16, 8), (0, 4))
        self.assertEqual(self._expected_range(20, 16, 4), (0, 4))
        self.assertEqual(self._expected_range(20, 16, 2), (-2, 6))

    def _check_foreground(self, labels, force_fg):
        for x in (2, 4, 8):
            dl, locations = self._loader(labels, x, p=1.0)
            for case_shape in self.case_shapes:
                for d in range(3):
                    lo, hi = self._expected_range(case_shape[d], self.final_patch_size[d], x)
                    for v in range(case_shape[d]):
                        voxel = [s // 2 for s in case_shape]
                        voxel[d] = v
                        locations.voxel = tuple(voxel)
                        start = self._final_patch_starts(dl, case_shape, force_fg)[d]
                        with self.subTest(x=x, case_size=case_shape[d], voxel=v):
                            # centered on the voxel as far as the range allows
                            self.assertEqual(start, min(hi, max(lo, v - self.final_patch_size[d] // 2)))
                            self.assertTrue(start <= v < start + self.final_patch_size[d])

    def test_oversampled_foreground(self):
        self._check_foreground({'background': 0, 'a': 1}, force_fg=True)

    def test_annotated_voxels_with_ignore_label(self):
        # with an ignore label, patches that are not oversampled are centered on an annotated voxel instead
        self._check_foreground({'background': 0, 'a': 1, 'ignore': 2}, force_fg=False)

    def test_2d_slice_axis_unchanged(self):
        # 2D configurations run as pseudo 3D with a final patch of 1 along the slice axis: any slice, as before
        np.random.seed(2)
        dl, _ = self._loader({'background': 0, 'a': 1}, 4, patch_size=(20, 28), final_patch_size=(16, 16), p=1.0)
        case_shape = (5, 10, 40)
        starts = np.array([self._final_patch_starts(dl, case_shape, False) for _ in range(2000)])
        self.assertEqual((starts[:, 0].min(), starts[:, 0].max()), (0, 4))
        self.assertEqual((starts[:, 1].min(), starts[:, 1].max()), self._expected_range(10, 16, 4))
        self.assertEqual((starts[:, 2].min(), starts[:, 2].max()), (0, 24))

    def test_invalid_arguments(self):
        for x in (0, -2):
            with self.assertRaises(ValueError):
                self._loader({'background': 0, 'a': 1}, x)
        for p in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                self._loader({'background': 0, 'a': 1}, 2, p=p)


class TestWithoutMinimumRange(_BboxTestCase):
    """
    min_patch_range_div=None: patches are placed only where the final patch lies inside the case, and a case smaller
    than the patch is centered. A foreground patch is centered on a foreground voxel and then moved into the range
    random patches are drawn from, at both borders.
    """

    def _final_patches(self, case_shape, labels, force_fg):
        """-> (axis, voxel coordinate along it, start and end of the final patch along it) for every voxel position
        along every axis, the other coordinates in the middle of the case"""
        dl, locations = self._loader(labels, None)
        self.assertEqual(dl.min_patch_range, [0, 0, 0])
        for d in range(3):
            for v in range(case_shape[d]):
                voxel = [s // 2 for s in case_shape]
                voxel[d] = v
                locations.voxel = tuple(voxel)
                start = self._final_patch_starts(dl, case_shape, force_fg)[d]
                yield d, v, start, start + self.final_patch_size[d]

    def _check(self, labels, force_fg):
        # axes 0 and 2 are larger than the final patch, axis 1 is smaller
        case_shape = (40, 10, 33)
        for d, v, start, end in self._final_patches(case_shape, labels, force_fg):
            with self.subTest(axis=d, voxel=v):
                self.assertTrue(start <= v < end, 'the foreground voxel must be in the final patch')
                if case_shape[d] >= self.final_patch_size[d]:
                    self.assertGreaterEqual(start, 0, 'final patch reaches below the case')
                    self.assertLessEqual(end, case_shape[d], 'final patch reaches above the case')
                else:
                    # a case smaller than the patch is padded on both sides, as for inference, wherever the
                    # foreground voxel is
                    pad_below, pad_above = -start, end - case_shape[d]
                    self.assertLessEqual(abs(pad_below - pad_above), 1, 'case is not centered in the patch')

    def test_oversampled_foreground(self):
        self._check({'background': 0, 'a': 1}, force_fg=True)

    def test_annotated_voxels_with_ignore_label(self):
        # with an ignore label, patches that are not oversampled are centered on an annotated voxel instead
        self._check({'background': 0, 'a': 1, 'ignore': 2}, force_fg=False)


class TestMinimumRangeProbability(_BboxTestCase):
    """
    min_patch_range_p: the minimum range is applied to a sample (along all axes at once) with that probability, and
    only to samples whose range it would widen; other samples are placed as with min_patch_range_div=None. No random
    number is drawn for that where nothing would be widened, or at p = 0 or 1.
    """
    # all three axes are widened at X = 2: smaller than the patch, exactly the patch, one voxel larger
    small_case = (10, 16, 17)
    # not widened at X = 2 along any axis: 24 positions of movement inside the case
    large_case = (40, 40, 40)

    @staticmethod
    def _inside_range(s, f):
        return (0, s - f) if s >= f else (-((f - s + 1) // 2), -((f - s) // 2))

    @staticmethod
    def _widened_range(s, f, x=2):
        lo, hi = TestMinimumRangeProbability._inside_range(s, f)
        deficit = f // x - (hi - lo)
        return (lo - deficit // 2, hi + deficit - deficit // 2) if deficit > 0 else (lo, hi)

    def _starts(self, dl, case_shape, force_fg=False, n=4000):
        return np.array([self._final_patch_starts(dl, case_shape, force_fg) for _ in range(n)])

    def _replay(self, dl, case_shape, rng, seed, n):
        """random bbox starts drawn as get_bbox draws them without a coin: one randint per axis over the given range
        of final patch starts"""
        np.random.seed(seed)
        margin = [(dl.patch_size[d] - dl.final_patch_size[d]) // 2 for d in range(3)]
        ranges = [rng(s, self.final_patch_size[d]) for d, s in enumerate(case_shape)]
        out = []
        for _ in range(n):
            out.append([np.random.randint(lo - margin[d], hi - margin[d] + 1) + margin[d]
                        for d, (lo, hi) in enumerate(ranges)])
        return out, np.random.randint(0, 2 ** 31)

    def _drawn(self, dl, case_shape, seed, n):
        np.random.seed(seed)
        return self._starts(dl, case_shape, n=n).tolist(), np.random.randint(0, 2 ** 31)

    def test_p0_keeps_patches_inside_and_p1_always_widens(self):
        np.random.seed(4)
        for p, rng in ((0.0, self._inside_range), (1.0, self._widened_range)):
            starts = self._starts(self._loader({'background': 0, 'a': 1}, 2, p=p)[0], self.small_case)
            for d in range(3):
                with self.subTest(p=p, axis=d):
                    self.assertEqual((starts[:, d].min(), starts[:, d].max()),
                                     rng(self.small_case[d], self.final_patch_size[d]))

    def test_half_of_the_samples_one_coin_per_sample(self):
        np.random.seed(5)
        starts = self._starts(self._loader({'background': 0, 'a': 1}, 2, p=0.5)[0], self.small_case)
        f = self.final_patch_size
        inside = [self._inside_range(s, f[d]) for d, s in enumerate(self.small_case)]
        widened = [self._widened_range(s, f[d]) for d, s in enumerate(self.small_case)]
        for d in range(3):
            self.assertEqual((starts[:, d].min(), starts[:, d].max()), widened[d])
        outside = np.zeros(len(starts), dtype=bool)
        for d in range(3):
            outside |= (starts[:, d] < inside[d][0]) | (starts[:, d] > inside[d][1])
        # a widened sample lands outside the inside range along some axis with probability
        # 1 - prod(positions inside / positions of the widened range) = 1 - 1/9 * 1/9 * 2/9
        p_any = 1 - np.prod([(i[1] - i[0] + 1) / (w[1] - w[0] + 1) for i, w in zip(inside, widened)])
        # one coin per sample: 0.5 * p_any (0.499); one coin per axis would give 0.81 here
        self.assertAlmostEqual(outside.mean(), 0.5 * p_any, delta=0.04)

    def test_no_coin_where_nothing_is_widened(self):
        # a case large enough along every axis draws exactly the random numbers it draws without the minimum range
        for p in (0.0, 0.5, 1.0):
            dl = self._loader({'background': 0, 'a': 1}, 2, p=p)[0]
            with self.subTest(p=p):
                self.assertEqual(self._drawn(dl, self.large_case, 6, 200),
                                 self._replay(dl, self.large_case, self._inside_range, 6, 200))

    def test_no_coin_at_p_0_or_1(self):
        for p, rng in ((1.0, self._widened_range), (0.0, self._inside_range)):
            dl = self._loader({'background': 0, 'a': 1}, 2, p=p)[0]
            with self.subTest(p=p):
                self.assertEqual(self._drawn(dl, self.small_case, 8, 200), self._replay(dl, self.small_case, rng, 8, 200))

    def test_foreground_patches(self):
        np.random.seed(7)
        f = self.final_patch_size
        for labels, force_fg in (({'background': 0, 'a': 1}, True), ({'background': 0, 'a': 1, 'ignore': 2}, False)):
            dl, locations = self._loader(labels, 2, p=0.5)
            for d, s in enumerate(self.small_case):
                (ilo, ihi), (wlo, whi) = self._inside_range(s, f[d]), self._widened_range(s, f[d])
                for v in range(s):
                    voxel = [x // 2 for x in self.small_case]
                    voxel[d] = v
                    locations.voxel = tuple(voxel)
                    starts = set(self._starts(dl, self.small_case, force_fg, n=60)[:, d].tolist())
                    centered = v - f[d] // 2
                    expected = {min(ihi, max(ilo, centered)), min(whi, max(wlo, centered))}
                    with self.subTest(force_fg=force_fg, axis=d, voxel=v):
                        # placed as without the minimum range or as with it, both occur, the voxel is in the patch
                        self.assertEqual(starts, expected)
                        self.assertTrue(all(st <= v < st + f[d] for st in starts))

if __name__ == '__main__':
    unittest.main()
