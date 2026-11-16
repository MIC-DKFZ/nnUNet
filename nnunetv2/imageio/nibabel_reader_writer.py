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
from nibabel.arrayproxy import ArrayProxy
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
from nibabel.volumeutils import apply_read_scaling

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

_IDENTITY_ORNT = [[0, 1], [1, 1], [2, 1]]
# voxels converted to float64 at a time when an image is read into the float32 result: 2 MB per temporary, small enough
# to stay in the CPU cache (one slice of a 512x512 CT; 4 MB blocks read it 20 % slower)
_BLOCK_VOXELS = 2 ** 18


def _reoriented_geometry(nib_image: nibabel.spatialimages.SpatialImage, ornt) -> nibabel.spatialimages.SpatialImage:
    """
    nib_image.as_reoriented(ornt) without its voxels: the same call on a zero-byte stand-in with the same shape, affine
    and header. Only the affine and header of the result are meaningful.
    """
    if np.array_equal(ornt, _IDENTITY_ORNT):
        return nib_image  # as_reoriented returns the image itself
    shape_only = np.broadcast_to(np.zeros((), dtype=np.uint8), nib_image.shape)
    return nib_image.__class__(shape_only, nib_image.affine, nib_image.header).as_reoriented(ornt)


def _read_into(out: np.ndarray, nib_image: nibabel.spatialimages.SpatialImage, ornt=None) -> None:
    """
    Writes the voxels of nib_image.get_fdata() (ornt None or identity), or of nib_image.as_reoriented(ornt).get_fdata(),
    transposed to SimpleITK's axis order and cast to float32, into out. These are exactly the voxels NibabelIO and
    NibabelIOWithReorient have always returned, but neither the float64 image those calls create nor a second copy of
    it is ever held: a stored image is read once in its stored dtype, reoriented as a view and converted block by block
    with nibabel's own scaling.
    """
    dataobj = nib_image.dataobj
    reorient = ornt is not None and not np.array_equal(ornt, _IDENTITY_ORNT)
    if type(dataobj) is ArrayProxy and (reorient or not nib_image.in_memory):
        # an image in a file (get_fdata() would return a filled cache instead; as_reoriented reads the file regardless).
        # ArrayProxy._get_scaled: get_fdata() scales in float64, as_reoriented (np.asanyarray(dataobj)) in the slope's
        # dtype, each where the scale factors cast to it safely
        array = dataobj.get_unscaled()
        slope, inter = np.asanyarray(dataobj.slope), np.asanyarray(dataobj.inter)
        scale_dtype = slope.dtype if reorient else np.dtype(np.float64)
        slope = slope.astype(scale_dtype) if np.can_cast(slope, scale_dtype) else slope
        inter = inter.astype(scale_dtype) if np.can_cast(inter, scale_dtype) else inter

        def to_float64(block):
            return np.asarray(apply_read_scaling(block, slope, inter), dtype=np.float64)
    elif isinstance(dataobj, np.ndarray) and (reorient or dataobj.dtype != np.float64):
        # an image in memory: get_fdata() is its array as float64 (or a filled cache of that, which this ignores)
        array = dataobj

        def to_float64(block):
            return np.asarray(block, dtype=np.float64)
    else:
        # float64 arrays, filled caches and other proxies: nibabel's float64 voxels as they are
        array = (nib_image.as_reoriented(ornt) if reorient else nib_image).get_fdata(caching='unchanged')
        reorient, to_float64 = False, None
    if reorient:
        array = apply_orientation(array, ornt)
    # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
    array = array.transpose((2, 1, 0))
    if to_float64 is None:
        out[...] = array
        return
    step = max(1, _BLOCK_VOXELS // max(1, array[0].size))
    for start in range(0, array.shape[0], step):
        out[start:start + step] = to_float64(array[start:start + step])


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
        return self.read_nibabel_images([nibabel.load(f) for f in image_fnames], image_fnames)

    def read_nibabel_images(self, nib_images: List[nibabel.Nifti1Image],
                            image_fnames: Union[List[str], Tuple[str, ...], None] = None) -> Tuple[np.ndarray, dict]:
        """read_images for images already in memory (e.g. handed over by another tool): exactly what read_images
        returns for the same images stored as files. image_fnames only names them in messages."""
        nib_images = list(nib_images)  # iterated twice: geometry first, then the voxels
        if image_fnames is None:
            image_fnames = [f'<in-memory image {i}>' for i in range(len(nib_images))]
        shapes = []
        original_affines = []

        spacings_for_nnunet = []
        for nib_image in nib_images:
            assert nib_image.ndim == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine

            original_affines.append(original_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                [float(i) for i in nib_image.header.get_zooms()[::-1]]
            )

            # the voxels are read once the geometry is checked, transposed to SimpleITK's axis order
            shapes.append((1, *nib_image.shape[::-1]))

        if not self._check_all_same(shapes):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print(shapes)
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
        data = np.empty((len(shapes), *shapes[0][1:]), dtype=np.float32)
        for channel, nib_image in zip(data, nib_images):
            _read_into(channel, nib_image)
        return data, dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        nibabel.save(self.seg_to_nibabel(seg, properties), output_fname)

    def seg_to_nibabel(self, seg: np.ndarray, properties: dict, name: str = 'in-memory segmentation') -> nibabel.Nifti1Image:
        """The image write_seg would write, in memory."""
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)
        return nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['original_affine'])


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
        return self.read_nibabel_images([nibabel.load(f) for f in image_fnames], image_fnames)

    def read_nibabel_images(self, nib_images: List[nibabel.Nifti1Image],
                            image_fnames: Union[List[str], Tuple[str, ...], None] = None) -> Tuple[np.ndarray, dict]:
        """read_images for images already in memory (e.g. handed over by another tool): exactly what read_images
        returns for the same images stored as files. image_fnames only names them in messages."""
        nib_images = list(nib_images)  # iterated twice: geometry first, then the voxels
        if image_fnames is None:
            image_fnames = [f'<in-memory image {i}>' for i in range(len(nib_images))]
        shapes = []
        orientations = []
        original_affines = []
        reoriented_affines = []

        spacings_for_nnunet = []
        for nib_image in nib_images:
            assert nib_image.ndim == 3, 'only 3d images are supported by NibabelIO'
            original_affine = nib_image.affine
            orientations.append(io_orientation(original_affine))
            reoriented_image = _reoriented_geometry(nib_image, orientations[-1])
            reoriented_affine = reoriented_image.affine

            original_affines.append(original_affine)
            reoriented_affines.append(reoriented_affine)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                [float(i) for i in reoriented_image.header.get_zooms()[::-1]]
            )

            # the voxels are read once the geometry is checked, transposed to SimpleITK's axis order
            shapes.append((1, *reoriented_image.shape[::-1]))

        if not self._check_all_same(shapes):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print(shapes)
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
        data = np.empty((len(shapes), *shapes[0][1:]), dtype=np.float32)
        for channel, nib_image, ornt in zip(data, nib_images, orientations):
            _read_into(channel, nib_image, ornt)
        return data, dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        nibabel.save(self.seg_to_nibabel(seg, properties, output_fname), output_fname)

    def seg_to_nibabel(self, seg: np.ndarray, properties: dict, name: str = 'in-memory segmentation') -> nibabel.Nifti1Image:
        """The image write_seg would write, in memory (reoriented back to the input's orientation)."""
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False)

        seg_nib = nibabel.Nifti1Image(seg, affine=properties['nibabel_stuff']['reoriented_affine'])
        # Solution from https://github.com/nipy/nibabel/issues/1063#issuecomment-967124057
        img_ornt = io_orientation(properties['nibabel_stuff']['original_affine'])
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)
        seg_nib_reoriented = seg_nib.as_reoriented(from_canonical)
        if not np.allclose(properties['nibabel_stuff']['original_affine'], seg_nib_reoriented.affine):
            print(f'WARNING: Restored affine does not match original affine. File: {name}')
            print('Original affine\n', properties['nibabel_stuff']['original_affine'])
            print('Restored affine\n', seg_nib_reoriented.affine)
        return seg_nib_reoriented


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
