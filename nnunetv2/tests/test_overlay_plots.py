import unittest

import numpy as np

from nnunetv2.utilities.overlay_plots import color_cycle, generate_overlay, hex_to_rgb, select_slice_to_plot2


class TestGenerateOverlay(unittest.TestCase):
    def test_more_labels_than_colors(self):
        # background and more foreground labels than there are colors in one slice
        num_labels = len(color_cycle) + 5
        seg = np.repeat(np.arange(num_labels), 3).reshape(num_labels, 3)
        image = np.linspace(0, 1, seg.size).reshape(seg.shape)
        overlay = generate_overlay(image, seg)
        self.assertEqual(overlay.shape, (*seg.shape, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_colors_wrap_around_without_background_color(self):
        n = len(color_cycle)
        # label: index in color_cycle. Labels 3 and 4 are beyond the end of the cycle
        mapping = {0: 0, 1: 1, 2: 2, 3: n, 4: n + 1, 5: n - 1}
        seg = np.array([[0, 1, 2, 3, 4, 5, 0]])
        # same intensity everywhere except for the last (background) pixel, so equal colors give equal pixels
        image = np.array([[0, 0, 0, 0, 0, 0, 1.]])
        overlay = generate_overlay(image, seg, mapping=mapping)
        # the background is not colored
        np.testing.assert_array_equal(overlay[0, 0], 0)
        # beyond the end of the cycle the colors continue with the first non-background one
        np.testing.assert_array_equal(overlay[0, 3], overlay[0, 1])
        np.testing.assert_array_equal(overlay[0, 4], overlay[0, 2])
        self.assertFalse(np.array_equal(overlay[0, 3], overlay[0, 0]))
        self.assertFalse(np.array_equal(overlay[0, 5], overlay[0, 1]))

    def test_constant_image_keeps_label_colors(self):
        # a slice of constant intensity (e.g. all zero) used to be rescaled with 0 / 0 = NaN, which turned the whole
        # overlay, labels included, into garbage (black after the uint8 cast)
        seg = np.zeros((4, 5), dtype=np.uint8)
        seg[1:3, 1:3] = 1
        for value in (0., 3.):
            with np.errstate(all='raise'):
                overlay = generate_overlay(np.full(seg.shape, value, dtype=np.float32), seg)
            np.testing.assert_array_equal(overlay[seg == 0], 0)
            # the only color in the overlay is rescaled to full intensity
            expected = np.array(hex_to_rgb(color_cycle[1]), dtype=float)
            np.testing.assert_array_equal(overlay[1, 1], (expected / expected.max() * 255).astype(np.uint8))

    def test_constant_image_without_labels(self):
        with np.errstate(all='raise'):
            overlay = generate_overlay(np.zeros((4, 5), dtype=np.int16), np.zeros((4, 5), dtype=np.uint8))
        np.testing.assert_array_equal(overlay, 0)

    def test_rgb_image(self):
        # the docstring promises 2d RGB input with the color channels in the last dimension
        rng = np.random.default_rng(0)
        image = rng.random((4, 5, 3))
        seg = np.zeros((4, 5), dtype=np.uint8)
        seg[0, 0] = 1
        overlay = generate_overlay(image, seg)
        self.assertEqual(overlay.shape, (4, 5, 3))
        # unlabeled pixels keep their colors (up to the rescaling to [0, 255])
        background = seg == 0
        expected = (image - image.min()) / (image - image.min()).max() * 255
        expected[0, 0] += 0.6 * np.array(hex_to_rgb(color_cycle[1]))
        expected = (expected / expected.max() * 255).astype(np.uint8)
        np.testing.assert_array_equal(overlay[background], expected[background])

    def test_image_with_wrong_number_of_color_channels(self):
        with self.assertRaises(RuntimeError):
            generate_overlay(np.zeros((4, 5, 2)), np.zeros((4, 5), dtype=np.uint8))


class TestSelectSliceToPlot2(unittest.TestCase):
    def test_small_class_is_not_underweighted(self):
        # class 1 has 60 % of its voxels in slice 0 and 40 % in slice 1. Class 2 is a single voxel in slice 2. Slice 2
        # holds 100 % of class 2, which beats the 60 % of class 1 in slice 0 (mean over classes 0.5 vs 0.3)
        seg = np.zeros((3, 10, 10), dtype=np.uint8)
        seg[0].flat[:60] = 1
        seg[1].flat[:40] = 1
        seg[2, 0, 0] = 2
        self.assertEqual(select_slice_to_plot2(np.zeros(seg.shape), seg), 2)

    def test_matches_per_class_fractions(self):
        rng = np.random.RandomState(1)
        seg = rng.choice(4, size=(12, 6, 6), p=(0.85, 0.1, 0.04, 0.01))
        classes = [c for c in np.unique(seg) if c > 0]
        expected = np.mean([(seg == c).sum((1, 2)) / (seg == c).sum() for c in classes], 0)
        self.assertEqual(select_slice_to_plot2(np.zeros(seg.shape), seg), int(np.argmax(expected)))


if __name__ == '__main__':
    unittest.main()
