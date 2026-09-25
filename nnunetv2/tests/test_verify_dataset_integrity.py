import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

from nnunetv2.experiment_planning.verify_dataset_integrity import check_cases
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


if __name__ == '__main__':
    unittest.main()
