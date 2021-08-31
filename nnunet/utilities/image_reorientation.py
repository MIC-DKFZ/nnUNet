#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

import nibabel as nib
from nibabel import io_orientation
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import os
from multiprocessing import Pool
import SimpleITK as sitk


def print_shapes(folder: str) -> None:
    for i in subfiles(folder, suffix='.nii.gz'):
        tmp = sitk.ReadImage(i)
        print(sitk.GetArrayFromImage(tmp).shape, tmp.GetSpacing())


def reorient_to_ras(image: str) -> None:
    """
    Will overwrite image!!!
    :param image:
    :return:
    """
    assert image.endswith('.nii.gz')
    origaffine_pkl = image[:-7] + '_originalAffine.pkl'
    if not isfile(origaffine_pkl):
        img = nib.load(image)
        original_affine = img.affine
        original_axcode = nib.aff2axcodes(img.affine)
        img = img.as_reoriented(io_orientation(img.affine))
        new_axcode = nib.aff2axcodes(img.affine)
        print(image.split('/')[-1], 'original axcode', original_axcode, 'now (should be ras)', new_axcode)
        nib.save(img, image)
        save_pickle((original_affine, original_axcode), origaffine_pkl)


def revert_reorientation(image: str) -> None:
    assert image.endswith('.nii.gz')
    expected_pkl = image[:-7] + '_originalAffine.pkl'
    assert isfile(expected_pkl), 'Must have a file with the original affine, as created by ' \
                                 'reorient_to_ras. Expected filename: %s' % \
                                 expected_pkl
    original_affine, original_axcode = load_pickle(image[:-7] + '_originalAffine.pkl')
    img = nib.load(image)
    before_revert = nib.aff2axcodes(img.affine)
    img = img.as_reoriented(io_orientation(original_affine))
    after_revert = nib.aff2axcodes(img.affine)
    print('before revert', before_revert, 'after revert', after_revert)
    restored_affine = img.affine
    assert np.all(np.isclose(original_affine, restored_affine)), 'restored affine does not match original affine, ' \
                                                                 'aborting!'
    nib.save(img, image)
    os.remove(expected_pkl)


def reorient_all_images_in_folder_to_ras(folder: str, num_processes: int = 8):
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(reorient_to_ras, nii_files)
    p.close()
    p.join()


def revert_orientation_on_all_images_in_folder(folder: str, num_processes: int = 8):
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(revert_reorientation, nii_files)
    p.close()
    p.join()


if __name__ == '__main__':
    """nib.as_closest_canonical()
    test_img = '/home/fabian/data/la_005_0000.nii.gz'
    test_img_reorient = test_img[:-7] + '_reorient.nii.gz'
    test_img_restored = test_img[:-7] + '_restored.nii.gz'

    img = nib.load(test_img)
    print('loaded original')
    print('shape', img.shape)
    print('affine', img.affine)
    original_affine = img.affine
    original_axcode = nib.aff2axcodes(img.affine)
    print('orientation', nib.aff2axcodes(img.affine))

    print('reorienting...')
    img = img.as_reoriented(io_orientation(img.affine))
    nib.save(img, test_img_reorient)

    print('now loading the reoriented img')
    img = nib.load(test_img_reorient)
    print('loaded original')
    print('shape', img.shape)
    print('affine', img.affine)
    reorient_affine = img.affine
    reorient_axcode = nib.aff2axcodes(img.affine)
    print('orientation', nib.aff2axcodes(img.affine))

    print('restoring original geometry')
    img = img.as_reoriented(io_orientation(original_affine))
    restored_affine = img.affine
    nib.save(img, test_img_restored)

    print('now loading the restored img')
    img = nib.load(test_img_restored)
    print('loaded original')
    print('shape', img.shape)
    print('affine', img.affine)
    print('orientation', nib.aff2axcodes(img.affine))"""
