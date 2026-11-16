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
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import SimpleITK as sitk


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha',
        '.gipl'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        return self.read_sitk_images([sitk.ReadImage(f) for f in image_fnames], image_fnames)

    def read_sitk_images(self, itk_images: List[sitk.Image],
                         image_fnames: Union[List[str], Tuple[str, ...], None] = None) -> Tuple[np.ndarray, dict]:
        """read_images for images already in memory (e.g. handed over by another tool): exactly what read_images
        returns for the same images stored as files. image_fnames only names them in messages."""
        if image_fnames is None:
            image_fnames = [f'<in-memory image {i}>' for i in range(len(itk_images))]
        images = []
        viewed_images = []  # GetArrayViewFromImage views an image's buffer: the image must outlive the vstack below
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f, itk_image in zip(image_fnames, itk_images):
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            viewed_images.append(itk_image)
            npy_image = sitk.GetArrayViewFromImage(itk_image)
            if npy_image.ndim == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return np.vstack(images, dtype=np.float32, casting='unsafe'), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        sitk.WriteImage(self.seg_to_sitk(seg, properties), output_fname, True)

    def seg_to_sitk(self, seg: np.ndarray, properties: dict) -> sitk.Image:
        """The image write_seg would write, in memory."""
        assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])
        return itk_image


def _dicom_orient_view(itk_image: sitk.Image, orientation: str) -> Tuple[np.ndarray, tuple, sitk.Image]:
    """
    sitk.DICOMOrient(itk_image, orientation) as (its voxels as GetArrayViewFromImage gives them, its (spacing, origin,
    direction), the image the voxels are a view of). For scalar 3d images the voxels are a view of itk_image's own
    buffer, permuted and flipped as DICOMOrient permutes and flips the axes, and the geometry is what DICOMOrient gives
    a one-byte stand-in with the same geometry. That avoids the reoriented copy (and DICOMOrient's intermediate one).
    """
    if itk_image.GetDimension() != 3 or itk_image.GetNumberOfComponentsPerPixel() != 1:
        oriented = sitk.DICOMOrient(itk_image, orientation)
        geometry = (oriented.GetSpacing(), oriented.GetOrigin(), oriented.GetDirection())
        return sitk.GetArrayViewFromImage(oriented), geometry, oriented
    stand_in = sitk.Image(itk_image.GetSize(), sitk.sitkUInt8)
    stand_in.CopyInformation(itk_image)
    oriented = sitk.DICOMOrient(stand_in, orientation)
    geometry = (oriented.GetSpacing(), oriented.GetOrigin(), oriented.GetDirection())
    del stand_in, oriented
    # DICOMOrient permutes the axes and flips some of them: each new direction column is +- one of the old ones
    dots = np.reshape(geometry[2], (3, 3)).T @ np.reshape(itk_image.GetDirection(), (3, 3))  # [new axis, old axis]
    source_axis = np.abs(dots).argmax(1)
    # numpy's axes are ITK's in reverse order
    view = sitk.GetArrayViewFromImage(itk_image).transpose([2 - source_axis[2 - i] for i in range(3)])
    for axis in range(3):
        if dots[axis, source_axis[axis]] < 0:
            view = np.flip(view, 2 - axis)
    return view, geometry, itk_image


class SimpleITKIOWithReorient(SimpleITKIO):
    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]], orientation = "RAS") -> Tuple[np.ndarray, dict]:
        return self.read_sitk_images([sitk.ReadImage(f) for f in image_fnames], image_fnames, orientation)

    def read_sitk_images(self, itk_images: List[sitk.Image], image_fnames: Union[List[str], Tuple[str, ...], None] = None,
                         orientation="RAS") -> Tuple[np.ndarray, dict]:
        """read_images for images already in memory (e.g. handed over by another tool): exactly what read_images
        returns for the same images stored as files. image_fnames only names them in messages."""
        if image_fnames is None:
            image_fnames = [f'<in-memory image {i}>' for i in range(len(itk_images))]
        images = []
        viewed_images = []  # GetArrayViewFromImage views an image's buffer: the image must outlive the vstack below
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f, itk_image in zip(image_fnames, itk_images):
            original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(itk_image.GetDirection())
            npy_image, geometry, viewed_image = _dicom_orient_view(itk_image, orientation)
            spacings.append(geometry[0])
            origins.append(geometry[1])
            directions.append(geometry[2])
            viewed_images.append(viewed_image)
            if npy_image.ndim == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif npy_image.ndim == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif npy_image.ndim == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0],
                'original_orientation': original_orientation
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        # the reoriented views can be permuted: write them into a C-ordered result, as np.vstack does for plain arrays
        data = np.empty((sum(i.shape[0] for i in images), *images[0].shape[1:]), dtype=np.float32)
        np.concatenate(images, axis=0, out=data, casting='unsafe')
        return data, dict

    def seg_to_sitk(self, seg: np.ndarray, properties: dict) -> sitk.Image:
        """The image write_seg would write, in memory (reoriented back to the input's orientation)."""
        return sitk.DICOMOrient(super().seg_to_sitk(seg, properties), properties['sitk_stuff']['original_orientation'])
