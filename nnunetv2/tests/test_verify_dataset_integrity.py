import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk
from skimage import io

from nnunetv2.experiment_planning.verify_dataset_integrity import check_cases
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


class TestCheckCasesSpacingDimensions(unittest.TestCase):
    def test_vector_segmentation_reports_incompatible_spacing_dimensions(self):
        with TemporaryDirectory() as tmp:
            image_file = Path(tmp) / 'image.nrrd'
            seg_file = Path(tmp) / 'mask.seg.nrrd'
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8)), str(image_file))
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 5, 6, 2), dtype=np.uint8), isVector=True), str(seg_file))

            output = StringIO()
            with redirect_stdout(output):
                valid = check_cases([str(image_file)], str(seg_file), 1, SimpleITKIO)

            self.assertFalse(valid)
            self.assertIn('different spacing dimensions', output.getvalue().lower())
            self.assertIn(str(seg_file), output.getvalue())

    def test_4d_scalar_segmentation_reports_physical_space_dimensions(self):
        with TemporaryDirectory() as tmp:
            image_file = Path(tmp) / 'image.nrrd'
            seg_file = Path(tmp) / 'mask.nrrd'
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8)), str(image_file))
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((2, 4, 5, 6), dtype=np.uint8), isVector=False),
                            str(seg_file))

            output = StringIO()
            with redirect_stdout(output):
                valid = check_cases([str(image_file)], str(seg_file), 1, SimpleITKIO)

            self.assertFalse(valid)
            self.assertIn('different origin dimensions', output.getvalue().lower())
            self.assertIn('different direction dimensions', output.getvalue().lower())
            self.assertIn('Direction images:', output.getvalue())
            self.assertIn('Direction seg:', output.getvalue())
            self.assertIn(str(seg_file), output.getvalue())

    def test_scalar_spacing_mismatch_still_reports_error(self):
        with TemporaryDirectory() as tmp:
            image_file = Path(tmp) / 'image.nrrd'
            seg_file = Path(tmp) / 'mask.nrrd'
            image = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8))
            seg = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8))
            seg.SetSpacing((1.2, 1.0, 1.0))
            sitk.WriteImage(image, str(image_file))
            sitk.WriteImage(seg, str(seg_file))

            output = StringIO()
            with redirect_stdout(output):
                valid = check_cases([str(image_file)], str(seg_file), 1, SimpleITKIO)

            self.assertFalse(valid)
            self.assertIn('Spacing mismatch', output.getvalue())

    def test_scalar_segmentation_still_passes(self):
        with TemporaryDirectory() as tmp:
            image_file = Path(tmp) / 'image.nrrd'
            seg_file = Path(tmp) / 'mask.nrrd'
            image = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8))
            seg = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8))
            sitk.WriteImage(image, str(image_file))
            sitk.WriteImage(seg, str(seg_file))

            self.assertTrue(check_cases([str(image_file)], str(seg_file), 1, SimpleITKIO))


class TestCheckCasesSegmentationChannels(unittest.TestCase):
    def test_rgb_png_segmentation_is_rejected_when_read(self):
        # label PNGs saved as RGB used to pass the integrity check and only crash during training
        with TemporaryDirectory() as tmp:
            image_file = Path(tmp) / 'case_0000.png'
            seg_file = Path(tmp) / 'case.png'
            io.imsave(str(image_file), np.zeros((5, 6), dtype=np.uint8), check_contrast=False)
            io.imsave(str(seg_file), np.zeros((5, 6, 3), dtype=np.uint8), check_contrast=False)

            with self.assertRaisesRegex(RuntimeError, 'has 3 channels'):
                NaturalImage2DIO().read_seg(str(seg_file))
            with self.assertRaisesRegex(RuntimeError, 'has 3 channels'):
                check_cases([str(image_file)], str(seg_file), 1, NaturalImage2DIO)

    def test_grayscale_png_segmentation_still_passes(self):
        with TemporaryDirectory() as tmp:
            image_file = Path(tmp) / 'case_0000.png'
            seg_file = Path(tmp) / 'case.png'
            io.imsave(str(image_file), np.zeros((5, 6, 3), dtype=np.uint8), check_contrast=False)
            seg = np.zeros((5, 6), dtype=np.uint8)
            seg[1:3, 2:4] = 1
            io.imsave(str(seg_file), seg, check_contrast=False)

            self.assertEqual(NaturalImage2DIO().read_seg(str(seg_file))[0].shape, (1, 1, 5, 6))
            self.assertTrue(check_cases([str(image_file)], str(seg_file), 3, NaturalImage2DIO))

    def test_multi_channel_segmentation_is_reported_for_any_reader(self):
        # SimpleITKIO returns a 4D scalar file as (c, x, y, z), so the segmentation has 2 channels here
        with TemporaryDirectory() as tmp:
            image_file = Path(tmp) / 'image.nrrd'
            seg_file = Path(tmp) / 'mask.nrrd'
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8)), str(image_file))
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((2, 4, 5, 6), dtype=np.uint8), isVector=False),
                            str(seg_file))

            output = StringIO()
            with redirect_stdout(output):
                valid = check_cases([str(image_file)], str(seg_file), 1, SimpleITKIO)

            self.assertFalse(valid)
            self.assertIn('single-channel integer label map but has 2 channels', output.getvalue())
            self.assertIn(str(seg_file), output.getvalue())


if __name__ == '__main__':
    unittest.main()
