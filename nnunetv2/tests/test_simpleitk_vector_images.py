"""
SimpleITKIO(WithReorient) reading multi-component (vector) images, e.g. RGB or multi-channel images saved with
isVector=True. GetArrayViewFromImage puts the components on the last axis; the readers must return them as channels
(first axis), with the spacing of the spatial axes, exactly as if each component had been saved as its own file.
"""
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO, SimpleITKIOWithReorient


def _save(array: np.ndarray, fname: str, spacing, is_vector: bool, direction=None) -> str:
    img = sitk.GetImageFromArray(array, isVector=is_vector)
    img.SetSpacing(spacing)
    if direction is not None:
        img.SetDirection(direction)
    sitk.WriteImage(img, fname)
    return fname


class TestSimpleITKVectorImages(unittest.TestCase):
    def _check(self, reader, spatial_shape, spacing, direction=None):
        num_components = 3
        rng = np.random.default_rng(0)
        components = rng.random((num_components, *spatial_shape)).astype(np.float32)
        with TemporaryDirectory() as tmp:
            vector_file = _save(np.moveaxis(components, 0, -1), join(tmp, 'vector.nii.gz'), spacing, True,
                                direction)
            scalar_files = [_save(c, join(tmp, f'scalar_{i}.nii.gz'), spacing, False, direction)
                            for i, c in enumerate(components)]
            self.assertEqual(sitk.ReadImage(vector_file).GetNumberOfComponentsPerPixel(), num_components)

            data, props = reader.read_images([vector_file])
            data_ref, props_ref = reader.read_images(scalar_files)

        self.assertEqual(data.shape, data_ref.shape)
        np.testing.assert_array_equal(data, data_ref)
        self.assertEqual(len(props['spacing']), 3)
        np.testing.assert_allclose(props['spacing'], props_ref['spacing'])

    def test_3d_vector_image(self):
        self._check(SimpleITKIO(), (5, 6, 7), (1., 2., 3.))

    def test_2d_vector_image(self):
        self._check(SimpleITKIO(), (6, 7), (0.5, 0.8))

    def test_3d_vector_image_reorient(self):
        # LPS storage: the reorientation to RAS flips two axes, which must act on the spatial axes, not the components
        self._check(SimpleITKIOWithReorient(), (5, 6, 7), (1., 2., 3.), (-1., 0, 0, 0, -1., 0, 0, 0, 1.))


if __name__ == '__main__':
    unittest.main()
