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

    def _loader(self, labels, x=_NOT_GIVEN, patch_size=None, final_patch_size=None):
        lm = LabelManager(labels, None)
        locations = _OneVoxelLocations(lm.annotated_classes_key)
        data = SimpleNamespace(identifiers=['case'], foreground_locations=locations)
        kwargs = {} if x is _NOT_GIVEN else {'min_patch_range_div': x}
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
            dl, _ = self._loader({'background': 0, 'a': 1}, x)
            for case_shape in self.case_shapes:
                starts = np.array([self._final_patch_starts(dl, case_shape, False) for _ in range(2000)])
                for d in range(3):
                    expected = self._expected_range(case_shape[d], self.final_patch_size[d], x)
                    with self.subTest(x=x, case_size=case_shape[d]):
                        self.assertEqual((starts[:, d].min(), starts[:, d].max()), expected)
                        # the point of it: never less movement than final_patch_size // x
                        self.assertGreaterEqual(expected[1] - expected[0], self.final_patch_size[d] // x)

    def test_default_is_half_the_final_patch(self):
        dl, _ = self._loader({'background': 0, 'a': 1})
        self.assertEqual(dl.min_patch_range_div, 2)
        self.assertEqual(dl.min_patch_range, [8, 8, 8])
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
            dl, locations = self._loader(labels, x)
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
        dl, _ = self._loader({'background': 0, 'a': 1}, 4, patch_size=(20, 28), final_patch_size=(16, 16))
        case_shape = (5, 10, 40)
        starts = np.array([self._final_patch_starts(dl, case_shape, False) for _ in range(2000)])
        self.assertEqual((starts[:, 0].min(), starts[:, 0].max()), (0, 4))
        self.assertEqual((starts[:, 1].min(), starts[:, 1].max()), self._expected_range(10, 16, 4))
        self.assertEqual((starts[:, 2].min(), starts[:, 2].max()), (0, 24))

    def test_invalid_divisor(self):
        for x in (0, -2):
            with self.assertRaises(ValueError):
                self._loader({'background': 0, 'a': 1}, x)


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


if __name__ == '__main__':
    unittest.main()
