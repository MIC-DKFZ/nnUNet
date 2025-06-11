#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import warnings
from typing import Tuple, Union, List
import numpy as np
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


class NibabelIO(BaseReaderWriter):
    """
    Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNetv2_plot_overlay_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.nii',
        '.nii.gz',
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert nib_image.ndim == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine

            original_affines.append(original_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                [float(i) for i in nib_image.header.get_zooms()[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(nib_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same_array(original_affines):
            print('WARNING! Not all input images have the same original_affines!')
            print('Affines:')
            print(original_affines)
            print('Image files:')
            print(image_fnames)
            print(
                'It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! This might be caused by them not '
                  'having the same affine')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        dict = {
            'nibabel_stuff': {
                'original_affine': original_affines[0],
            },
            'spacing': spacings_for_nnunet[0]
        }
        return np.vstack(images, dtype=np.float32, casting='unsafe'), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)
        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['original_affine'])
        nibabel.save(seg_nib, output_fname)


class NibabelIOWithReorient(BaseReaderWriter):
    """
    Reorients images to RAS

    Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNetv2_plot_overlay_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.nii',
        '.nii.gz',
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []
        reoriented_affines = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert nib_image.ndim == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine
            reoriented_image = nib_image.as_reoriented(io_orientation(original_affine))
            reoriented_affine = reoriented_image.affine

            original_affines.append(original_affine)
            reoriented_affines.append(reoriented_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                [float(i) for i in reoriented_image.header.get_zooms()[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(reoriented_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same_array(reoriented_affines):
            print('WARNING! Not all input images have the same reoriented_affines!')
            print('Affines:')
            print(reoriented_affines)
            print('Image files:')
            print(image_fnames)
            print(
                'It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! This might be caused by them not '
                  'having the same affine')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        dict = {
            'nibabel_stuff': {
                'original_affine': original_affines[0],
                'reoriented_affine': reoriented_affines[0],
            },
            'spacing': spacings_for_nnunet[0]
        }
        return np.vstack(images, dtype=np.float32, casting='unsafe'), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)

        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['reoriented_affine'])
        # Solution from https://github.com/nipy/nibabel/issues/1063#issuecomment-967124057
        img_ornt = io_orientation(properties['nibabel_stuff']['original_affine'])
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)
        seg_nib_reoriented = seg_nib.as_reoriented(from_canonical)
        if not np.allclose(properties['nibabel_stuff']['original_affine'], seg_nib_reoriented.affine):
            print(f'WARNING: Restored affine does not match original affine. File: {output_fname}')
            print(f'Original affine\n', properties['nibabel_stuff']['original_affine'])
            print(f'Restored affine\n', seg_nib_reoriented.affine)
        nibabel.save(seg_nib_reoriented, output_fname)


if __name__ == '__main__':
    img_file = '/media/isensee/raw_data/nnUNet_raw/Dataset220_KiTS2023/imagesTr/case_00004_0000.nii.gz'
    seg_file = '/media/isensee/raw_data/nnUNet_raw/Dataset220_KiTS2023/labelsTr/case_00004.nii.gz'

    nibio = NibabelIO()
    # images, dct = nibio.read_images([img_file])
    seg, dctseg = nibio.read_seg(seg_file)

    nibio_r = NibabelIOWithReorient()
    # images_r, dct_r = nibio_r.read_images([img_file])
    seg_r, dctseg_r = nibio_r.read_seg(seg_file)

    sitkio = SimpleITKIO()
    # images_sitk, dct_sitk = sitkio.read_images([img_file])
    seg_sitk, dctseg_sitk = sitkio.read_seg(seg_file)

    # write reoriented and original segmentation
    nibio.write_seg(seg[0], '/home/isensee/seg_nibio.nii.gz', dctseg)
    nibio_r.write_seg(seg_r[0], '/home/isensee/seg_nibio_r.nii.gz', dctseg_r)
    sitkio.write_seg(seg_sitk[0], '/home/isensee/seg_nibio_sitk.nii.gz', dctseg_sitk)

    # now load all with sitk to make sure no shaped got f'd up
    a, d1 = sitkio.read_seg('/home/isensee/seg_nibio.nii.gz')
    b, d2 = sitkio.read_seg('/home/isensee/seg_nibio_r.nii.gz')
    c, d3 = sitkio.read_seg('/home/isensee/seg_nibio_sitk.nii.gz')

    assert a.shape == b.shape
    assert b.shape == c.shape

    assert np.all(a == b)
    assert np.all(b == c)
