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

from typing import Tuple, Union, List
import numpy as np
from nibabel import io_orientation

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel


class NibabelIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def __init__(self):
        """
        Reorients images to RAS

        Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
        consistent. This is of course considered properly in segmentation export as well.

        IMPORTANT: Run nnUNet_plot_task_pngs to verify that this did not destroy the alignment of data and seg!
        """
        super().__init__()

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []
        reoriented_affines = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert len(nib_image.shape) == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine
            reoriented_image = nib_image.as_reoriented(io_orientation(original_affine))
            reoriented_affine = reoriented_image.affine

            original_affines.append(original_affine)
            reoriented_affines.append(reoriented_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append((
                reoriented_affine[2, 2],
                reoriented_affine[1, 1],
                reoriented_affine[0, 0],
            ))

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
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_task_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            'nibabel_stuff': {
                'original_affine': original_affines[0],
                'reoriented_affine': reoriented_affines[0],
            },
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, image_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((image_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8)

        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['reoriented_affine'])
        seg_nib_reoriented = seg_nib.as_reoriented(io_orientation(properties['nibabel_stuff']['original_affine']))
        assert np.all(np.isclose(properties['nibabel_stuff']['original_affine'], seg_nib_reoriented.affine)), \
            'restored affine does not match original affine'
        nibabel.save(seg_nib_reoriented, output_fname)
