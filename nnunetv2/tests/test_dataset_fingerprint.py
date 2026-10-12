import unittest

import numpy as np

from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor


class TestDatasetFingerprintExtractor(unittest.TestCase):
    def test_collect_foreground_intensities_excludes_ignore_label(self):
        # issue #3055: voxels carrying the ignore label are not foreground and must not
        # be part of the foreground intensity statistics
        rng = np.random.RandomState(0)
        images = rng.uniform(0, 1, size=(1, 16, 16, 16)).astype(np.float32)
        segmentation = np.zeros((1, 16, 16, 16), dtype=np.uint8)
        segmentation[0, :8] = 1  # foreground
        segmentation[0, 8:] = 2  # ignore label, with very different intensities
        images[0, 8:] = 1000.0

        _, stats_with_ignore = DatasetFingerprintExtractor.collect_foreground_intensities(
            segmentation, images, num_samples=1000, ignore_label=2)
        _, stats_without_ignore = DatasetFingerprintExtractor.collect_foreground_intensities(
            segmentation, images, num_samples=1000)

        expected_mean = float(np.mean(images[0][segmentation[0] == 1]))
        # with the ignore label excluded, only the true foreground contributes
        self.assertAlmostEqual(stats_with_ignore[0]['mean'], expected_mean, places=2)
        self.assertLess(stats_with_ignore[0]['max'], 2.0)
        self.assertLess(stats_with_ignore[0]['percentile_99_5'], 2.0)

        # without passing an ignore label the previous (buggy) behavior is retained:
        # the ignore voxels are part of the foreground mask
        self.assertAlmostEqual(stats_without_ignore[0]['mean'],
                               float(np.mean(images[0][segmentation[0] > 0])), places=2)
        self.assertAlmostEqual(stats_without_ignore[0]['max'], 1000.0, places=4)

    def test_collect_foreground_intensities_without_ignore_label_in_segmentation(self):
        # passing an ignore label that is not present in the segmentation changes nothing
        rng = np.random.RandomState(42)
        images = rng.uniform(0, 1, size=(1, 8, 8, 8)).astype(np.float32)
        segmentation = np.zeros((1, 8, 8, 8), dtype=np.uint8)
        segmentation[0, :4] = 1

        _, stats_a = DatasetFingerprintExtractor.collect_foreground_intensities(
            segmentation, images, num_samples=100, ignore_label=5)
        _, stats_b = DatasetFingerprintExtractor.collect_foreground_intensities(
            segmentation, images, num_samples=100)

        self.assertAlmostEqual(stats_a[0]['mean'], stats_b[0]['mean'], places=6)
        self.assertEqual(stats_a[0]['min'], stats_b[0]['min'])
        self.assertEqual(stats_a[0]['max'], stats_b[0]['max'])


if __name__ == '__main__':
    unittest.main()
