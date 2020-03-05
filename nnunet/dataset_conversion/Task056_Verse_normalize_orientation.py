"""
This code is copied from https://gist.github.com/nlessmann/24d405eaa82abba6676deb6be839266c. All credits go to the
original author (user nlessmann on GitHub)
"""

import numpy as np
import SimpleITK as sitk


def reverse_axes(image):
    return np.transpose(image, tuple(reversed(range(image.ndim))))


def read_image(imagefile):
    image = sitk.ReadImage(imagefile)
    data = reverse_axes(sitk.GetArrayFromImage(image))  # switch from zyx to xyz
    header = {
        'spacing': image.GetSpacing(),
        'origin': image.GetOrigin(),
        'direction': image.GetDirection()
    }
    return data, header


def save_image(img: np.ndarray, header: dict, output_file: str):
    """
    CAREFUL you need to restore_original_slice_orientation before saving!
    :param img:
    :param header:
    :return:
    """
    # reverse back
    img = reverse_axes(img)  # switch from zyx to xyz
    img_itk = sitk.GetImageFromArray(img)
    img_itk.SetSpacing(header['spacing'])
    img_itk.SetOrigin(header['origin'])
    if not isinstance(header['direction'], tuple):
        img_itk.SetDirection(header['direction'].flatten())
    else:
        img_itk.SetDirection(header['direction'])

    sitk.WriteImage(img_itk, output_file)


def swap_flip_dimensions(cosine_matrix, image, header=None):
    # Compute swaps and flips
    swap = np.argmax(abs(cosine_matrix), axis=0)
    flip = np.sum(cosine_matrix, axis=0)

    # Apply transformation to image volume
    image = np.transpose(image, tuple(swap))
    image = image[tuple(slice(None, None, int(f)) for f in flip)]

    if header is None:
        return image

    # Apply transformation to header
    header['spacing'] = tuple(header['spacing'][s] for s in swap)
    header['direction'] = np.eye(3)

    return image, header


def normalize_slice_orientation(image, header):
    # Preserve original header so that we can easily transform back
    header['original'] = header.copy()

    # Compute inverse of cosine (round first because we assume 0/1 values only)
    # to determine how the image has to be transposed and flipped for cosine = identity
    cosine = np.asarray(header['direction']).reshape(3, 3)
    cosine_inv = np.linalg.inv(np.round(cosine))

    return swap_flip_dimensions(cosine_inv, image, header)


def restore_original_slice_orientation(mask, header):
    # Use original orientation for transformation because we assume the image to be in
    # normalized orientation, i.e., identity cosine)
    cosine = np.asarray(header['original']['direction']).reshape(3, 3)
    cosine_rnd = np.round(cosine)

    # Apply transformations to both the image and the mask
    return swap_flip_dimensions(cosine_rnd, mask), header['original']
