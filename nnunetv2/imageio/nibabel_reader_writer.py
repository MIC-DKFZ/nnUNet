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
from nibabel import io_orientation

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel


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
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
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
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8)
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
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
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
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8, copy=False)

        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['reoriented_affine'])
        seg_nib_reoriented = seg_nib.as_reoriented(io_orientation(properties['nibabel_stuff']['original_affine']))
        if not np.allclose(properties['nibabel_stuff']['original_affine'], seg_nib_reoriented.affine):
            print(f'WARNING: Restored affine does not match original affine. File: {output_fname}')
            print(f'Original affine\n', properties['nibabel_stuff']['original_affine'])
            print(f'Restored affine\n', seg_nib_reoriented.affine)
        nibabel.save(seg_nib_reoriented, output_fname)


if __name__ == '__main__':
    img_file = 'patient028_frame01_0000.nii.gz'
    seg_file = 'patient028_frame01.nii.gz'

    nibio = NibabelIO()
    images, dct = nibio.read_images([img_file])
    seg, dctseg = nibio.read_seg(seg_file)

    nibio_r = NibabelIOWithReorient()
    images_r, dct_r = nibio_r.read_images([img_file])
    seg_r, dctseg_r = nibio_r.read_seg(seg_file)

    nibio.write_seg(seg[0], '/home/isensee/seg_nibio.nii.gz', dctseg)
    nibio_r.write_seg(seg_r[0], '/home/isensee/seg_nibio_r.nii.gz', dctseg_r)

    s_orig = nibabel.load(seg_file).get_fdata()
    s_nibio = nibabel.load('/home/isensee/seg_nibio.nii.gz').get_fdata()
    s_nibio_r = nibabel.load('/home/isensee/seg_nibio_r.nii.gz').get_fdata()
