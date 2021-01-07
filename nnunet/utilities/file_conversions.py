from typing import Tuple, List
from skimage import io
import SimpleITK as sitk
import numpy as np
import tifffile


def convert_2d_image_to_nifti(input_filename: str, output_filename_truncated: str, spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Reads an image (must be a format that it recognized by skimage.io.imread) and converts it into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!

    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net

    If Transform is not None it will be applied to the image after loading.

    :param is_seg:
    :param transform:
    :param input_filename:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    img = io.imread(input_filename)

    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

    for j, i in enumerate(img):
        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        if not is_seg:
            sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")


def convert_3d_tiff_to_nifti(filenames: List[str], output_name: str, spacing: Tuple[tuple, list], transform=None, is_seg=False) -> None:
    """
    filenames must be a list of strings, each pointing to a separate 3d tiff file. One file per modality. If your data
    only has one imaging modality, simply pass a list with only a single entry

    Files in filenames must be readable with

    Note: we always only pass one file into tifffile.imread, not multiple (even though it supports it). This is because
    I am not familiar enough with this functionality and would like to have control over what happens.

    If Transform is not None it will be applied to the image after loading.

    :param transform:
    :param filenames:
    :param output_name:
    :param spacing:
    :return:
    """
    if is_seg:
        assert len(filenames) == 1

    for j, i in enumerate(filenames):
        img = tifffile.imread(i)

        if transform is not None:
            img = transform(img)

        itk_img = sitk.GetImageFromArray(img)
        itk_img.SetSpacing(list(spacing)[::-1])

        if not is_seg:
            sitk.WriteImage(itk_img, output_name + "_%04.0d.nii.gz" % j)
        else:
            sitk.WriteImage(itk_img, output_name + ".nii.gz")


def convert_2d_segmentation_nifti_to_img(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert img.shape[0] == 1, "This function can only export 2D segmentations!"
    img = img[0]
    if transform is not None:
        img = transform(img)

    io.imsave(output_filename, img.astype(export_dtype), check_contrast=False)


def convert_3d_segmentation_nifti_to_tiff(nifti_file: str, output_filename: str, transform=None, export_dtype=np.uint8):
    img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_file))
    assert len(img.shape) == 3, "This function can only export 3D segmentations!"
    if transform is not None:
        img = transform(img)

    tifffile.imsave(output_filename, img.astype(export_dtype))
