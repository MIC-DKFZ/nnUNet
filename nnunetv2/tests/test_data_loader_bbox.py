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


class TestForegroundPatchesStayInsideTheCase(unittest.TestCase):
    """
    A foreground patch is centered on a foreground voxel and then moved into the range random patches are drawn
    from. It used to be moved at the lower border only, so near the upper border up to half the final patch was
    padding, while random patches and foreground patches at the lower border never contained any.
    """
    final_patch_size = (16, 16, 16)
    # the initial patch is larger so that rotations do not pull in padding; the final patch is its center
    patch_size = (24, 20, 28)

    def _loader(self, labels):
        lm = LabelManager(labels, None)
        locations = _OneVoxelLocations(lm.annotated_classes_key)
        data = SimpleNamespace(identifiers=['case'], foreground_locations=locations)
        return nnUNetDataLoader(data, 1, self.patch_size, self.final_patch_size, lm, 1.0), locations

    def _final_patches(self, case_shape, labels, force_fg):
        """-> (axis, voxel coordinate along it, start and end of the final patch along it) for every voxel position
        along every axis, the other coordinates in the middle of the case"""
        dl, locations = self._loader(labels)
        for d in range(3):
            for v in range(case_shape[d]):
                voxel = [s // 2 for s in case_shape]
                voxel[d] = v
                locations.voxel = tuple(voxel)
                bbox_lbs, bbox_ubs = dl.get_bbox('case', np.array(case_shape), force_fg)
                self.assertEqual(bbox_ubs[d] - bbox_lbs[d], self.patch_size[d])
                start = bbox_lbs[d] + (self.patch_size[d] - self.final_patch_size[d]) // 2
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
