import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIOWithReorient

# LAS storage: index x runs towards the patient's left, y towards anterior, z towards superior
LAS_DIRECTION = (1., 0., 0., 0., -1., 0., 0., 0., 1.)


def _write_las_image(fname: str, array_zyx: np.ndarray) -> None:
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing((0.8, 0.9, 2.5))
    image.SetOrigin((-10., 20., -30.))
    image.SetDirection(LAS_DIRECTION)
    sitk.WriteImage(image, fname)


class TestSimpleITKIOWithReorientWriteSeg(unittest.TestCase):
    def _roundtrip(self, max_label: int):
        """
        Reads an LAS-stored image (the reader reorients it to RAS), writes the array back as a segmentation and returns
        (expected labels in the file's own storage order, the written image).
        """
        shape = (4, 5, 6)
        n = int(np.prod(shape))
        # labels span 0 ... max_label
        labels = (np.arange(n) * max_label // (n - 1)).reshape(shape).astype(np.int16)
        self.assertEqual(labels.max(), max_label)
        with TemporaryDirectory() as tmp:
            image_file = os.path.join(tmp, 'image.nii.gz')
            seg_file = os.path.join(tmp, 'seg.nii.gz')
            _write_las_image(image_file, labels)
            rw = SimpleITKIOWithReorient()
            data, properties = rw.read_images([image_file])
            self.assertEqual(properties['sitk_stuff']['original_orientation'], 'LAS')
            # the reader returns RAS arrays, so the stored order must differ from what was written
            self.assertFalse(np.array_equal(data[0], labels))
            rw.write_seg(data[0].astype(np.int32), seg_file, properties)
            written = sitk.ReadImage(seg_file)
            written_array = sitk.GetArrayFromImage(written)
        return labels, written, written_array

    def test_labels_below_255_are_uint8_and_restore_orientation(self):
        labels, written, written_array = self._roundtrip(max_label=125)
        self.assertEqual(written_array.dtype, np.uint8)
        self.assertEqual(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(written.GetDirection()), 'LAS')
        np.testing.assert_allclose(written.GetDirection(), LAS_DIRECTION)
        np.testing.assert_allclose(written.GetOrigin(), (-10., 20., -30.))
        np.testing.assert_allclose(written.GetSpacing(), (0.8, 0.9, 2.5))
        np.testing.assert_array_equal(written_array, labels)

    def test_labels_at_or_above_255_are_uint16(self):
        labels, written, written_array = self._roundtrip(max_label=300)
        self.assertEqual(written_array.dtype, np.uint16)
        np.testing.assert_array_equal(written_array, labels)


if __name__ == '__main__':
    unittest.main()
