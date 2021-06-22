# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from builtins import range, zip
import random

import numpy
import numpy as np
from copy import deepcopy
from scipy.ndimage import map_coordinates, fourier_gaussian
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage.morphology import grey_dilation
from skimage.transform import resize
from scipy.ndimage.measurements import label as lb


def generate_elastic_transform_coordinates(shape, alpha, sigma):
    n_dim = len(shape)
    offsets = []
    for _ in range(n_dim):
        offsets.append(gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.meshgrid(*tmp, indexing='ij')
    indices = [np.reshape(i + j, (-1, 1)) for i, j in zip(offsets, coords)]
    return indices


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords


def convert_seg_image_to_one_hot_encoding(image, classes=None):
    '''
    image must be either (x, y, z) or (x, y)
    Takes as input an nd array of a label map (any dimension). Outputs a one hot encoding of the label map.
    Example (3D): if input is of shape (x, y, z), the output will ne of shape (n_classes, x, y, z)
    '''
    if classes is None:
        classes = np.unique(image)
    out_image = np.zeros([len(classes)]+list(image.shape), dtype=image.dtype)
    for i, c in enumerate(classes):
        out_image[i][image == c] = 1
    return out_image


def convert_seg_image_to_one_hot_encoding_batched(image, classes=None):
    '''
    same as convert_seg_image_to_one_hot_encoding, but expects image to be (b, x, y, z) or (b, x, y)
    '''
    if classes is None:
        classes = np.unique(image)
    output_shape = [image.shape[0]] + [len(classes)] + list(image.shape[1:])
    out_image = np.zeros(output_shape, dtype=image.dtype)
    for b in range(image.shape[0]):
        for i, c in enumerate(classes):
            out_image[b, i][image[b] == c] = 1
    return out_image


def elastic_deform_coordinates(coordinates, alpha, sigma):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices


def elastic_deform_coordinates_2(coordinates, sigmas, magnitudes):
    '''
    magnitude can be a tuple/list
    :param coordinates:
    :param sigma:
    :param magnitude:
    :return:
    '''
    if not isinstance(magnitudes, (tuple, list)):
        magnitudes = [magnitudes] * (len(coordinates) - 1)
    if not isinstance(sigmas, (tuple, list)):
        sigmas = [sigmas] * (len(coordinates) - 1)
    n_dim = len(coordinates)
    offsets = []
    for d in range(n_dim):
        random_values = np.random.random(coordinates.shape[1:]) * 2 - 1
        random_values_ = numpy.fft.fftn(random_values)
        deformation_field = fourier_gaussian(random_values_, sigmas)
        deformation_field = numpy.fft.ifftn(deformation_field).real
        offsets.append(deformation_field)
        mx = np.max(np.abs(offsets[-1]))
        offsets[-1] = offsets[-1] / (mx / (magnitudes[d] + 1e-8))
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def rotate_coords_2d(coords, angle):
    rot_matrix = create_matrix_rotation_2d(angle)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def scale_coords(coords, scale):
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        for i in range(len(scale)):
            coords[i] *= scale[i]
    else:
        coords *= scale
    return coords


def uncenter_coords(coords):
    shp = coords.shape[1:]
    coords = deepcopy(coords)
    for d in range(coords.shape[0]):
        coords[d] += (shp[d] - 1) / 2.
    return coords


def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img == c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)


def generate_noise(shape, alpha, sigma):
    noise = np.random.random(shape) * 2 - 1
    noise = gaussian_filter(noise, sigma, mode="constant", cval=0) * alpha
    return noise


def find_entries_in_array(entries, myarray):
    entries = np.array(entries)
    values = np.arange(np.max(myarray) + 1)
    lut = np.zeros(len(values), 'bool')
    lut[entries.astype("int")] = True
    return np.take(lut, myarray.astype(int))


def center_crop_3D_image(img, crop_size):
    center = np.array(img.shape) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * len(img.shape)
    else:
        center_crop = crop_size
        assert len(center_crop) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.),
           int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]


def center_crop_3D_image_batched(img, crop_size):
    # dim 0 is batch, dim 1 is channel, dim 2, 3 and 4 are x y z
    center = np.array(img.shape[2:]) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * (len(img.shape) - 2)
    else:
        center_crop = crop_size
        assert len(center_crop) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    return img[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.),
           int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]


def center_crop_2D_image(img, crop_size):
    center = np.array(img.shape) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * len(img.shape)
    else:
        center_crop = crop_size
        assert len(center_crop) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]


def center_crop_2D_image_batched(img, crop_size):
    # dim 0 is batch, dim 1 is channel, dim 2 and 3 are x y
    center = np.array(img.shape[2:]) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * (len(img.shape) - 2)
    else:
        center_crop = crop_size
        assert len(center_crop) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]


def random_crop_3D_image(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[2]:
        lb_z = np.random.randint(0, img.shape[2] - crop_size[2])
    elif crop_size[2] == img.shape[2]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")

    return img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]


def random_crop_3D_image_batched(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[4]:
        lb_z = np.random.randint(0, img.shape[4] - crop_size[2])
    elif crop_size[2] == img.shape[4]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")

    return img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]


def random_crop_2D_image(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"

    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    return img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]


def random_crop_2D_image_batched(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    return img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]


def resize_image_by_padding(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
    elif len(shape) == 3:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image
    return res


def resize_image_by_padding_batched(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape[2:]))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res = np.ones((image.shape[0], image.shape[1], new_shape[0], new_shape[1]), dtype=image.dtype) * pad_value
        res[:, :, int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image[:,
                                                                                                              :]
    elif len(shape) == 3:
        res = np.ones((image.shape[0], image.shape[1], new_shape[0], new_shape[1], new_shape[2]),
                      dtype=image.dtype) * pad_value
        res[:, :, int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image[:, :]
    else:
        raise RuntimeError("unexpected dimension")
    return res


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def create_matrix_rotation_2d(angle, matrix=None):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation

    return np.dot(matrix, rotation)


def create_random_rotation(angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi)):
    return create_matrix_rotation_x_3d(np.random.uniform(*angle_x),
                                       create_matrix_rotation_y_3d(np.random.uniform(*angle_y),
                                                                   create_matrix_rotation_z_3d(
                                                                       np.random.uniform(*angle_z))))


def illumination_jitter(img, u, s, sigma):
    # img must have shape [....., c] where c is the color channel
    alpha = np.random.normal(0, sigma, s.shape)
    jitter = np.dot(u, alpha * s)
    img2 = np.array(img)
    for c in range(img.shape[0]):
        img2[c] = img[c] + jitter[c]
    return img2


def general_cc_var_num_channels(img, diff_order=0, mink_norm=1, sigma=1, mask_im=None, saturation_threshold=255,
                                dilation_size=3, clip_range=True):
    # img must have first dim color channel! img[c, x, y(, z, ...)]
    dim_img = len(img.shape[1:])
    if clip_range:
        minm = img.min()
        maxm = img.max()
    img_internal = np.array(img)
    if mask_im is None:
        mask_im = np.zeros(img_internal.shape[1:], dtype=bool)
    img_dil = deepcopy(img_internal)
    for c in range(img.shape[0]):
        img_dil[c] = grey_dilation(img_internal[c], tuple([dilation_size] * dim_img))
    mask_im = mask_im | np.any(img_dil >= saturation_threshold, axis=0)
    if sigma != 0:
        mask_im[:sigma, :] = 1
        mask_im[mask_im.shape[0] - sigma:, :] = 1
        mask_im[:, mask_im.shape[1] - sigma:] = 1
        mask_im[:, :sigma] = 1
        if dim_img == 3:
            mask_im[:, :, mask_im.shape[2] - sigma:] = 1
            mask_im[:, :, :sigma] = 1

    output_img = deepcopy(img_internal)

    if diff_order == 0 and sigma != 0:
        for c in range(img_internal.shape[0]):
            img_internal[c] = gaussian_filter(img_internal[c], sigma, diff_order)
    elif diff_order == 1:
        for c in range(img_internal.shape[0]):
            img_internal[c] = gaussian_gradient_magnitude(img_internal[c], sigma)
    elif diff_order > 1:
        raise ValueError("diff_order can only be 0 or 1. 2 is not supported (ToDo, maybe)")

    img_internal = np.abs(img_internal)

    white_colors = []

    if mink_norm != -1:
        kleur = np.power(img_internal, mink_norm)
        for c in range(kleur.shape[0]):
            white_colors.append(np.power((kleur[c][mask_im != 1]).sum(), 1. / mink_norm))
    else:
        for c in range(img_internal.shape[0]):
            white_colors.append(np.max(img_internal[c][mask_im != 1]))

    som = np.sqrt(np.sum([i ** 2 for i in white_colors]))

    white_colors = [i / som for i in white_colors]

    for c in range(output_img.shape[0]):
        output_img[c] /= (white_colors[c] * np.sqrt(3.))

    if clip_range:
        output_img[output_img < minm] = minm
        output_img[output_img > maxm] = maxm
    return white_colors, output_img


def convert_seg_to_bounding_box_coordinates(data_dict, dim, get_rois_from_seg_flag=False, class_specific_seg_flag=False):

        '''
        This function generates bounding box annotations from given pixel-wise annotations.
        :param data_dict: Input data dictionary as returned by the batch generator.
        :param dim: Dimension in which the model operates (2 or 3).
        :param get_rois_from_seg: Flag specifying one of the following scenarios:
        1. A label map with individual ROIs identified by increasing label values, accompanied by a vector containing
        in each position the class target for the lesion with the corresponding label (set flag to False)
        2. A binary label map. There is only one foreground class and single lesions are not identified.
        All lesions have the same class target (foreground). In this case the Dataloader runs a Connected Component
        Labelling algorithm to create processable lesion - class target pairs on the fly (set flag to True).
        :param class_specific_seg_flag: if True, returns the pixelwise-annotations in class specific manner,
        e.g. a multi-class label map. If False, returns a binary annotation map (only foreground vs. background).
        :return: data_dict: same as input, with additional keys:
        - 'bb_target': bounding box coordinates (b, n_boxes, (y1, x1, y2, x2, (z1), (z2)))
        - 'roi_labels': corresponding class labels for each box (b, n_boxes, class_label)
        - 'roi_masks': corresponding binary segmentation mask for each lesion (box). Only used in Mask RCNN. (b, n_boxes, y, x, (z))
        - 'seg': now label map (see class_specific_seg_flag)
        '''

        bb_target = []
        roi_masks = []
        roi_labels = []
        out_seg = np.copy(data_dict['seg'])
        for b in range(data_dict['seg'].shape[0]):

            p_coords_list = []
            p_roi_masks_list = []
            p_roi_labels_list = []

            if np.sum(data_dict['seg'][b]!=0) > 0:
                if get_rois_from_seg_flag:
                    clusters, n_cands = lb(data_dict['seg'][b])
                    data_dict['class_target'][b] = [data_dict['class_target'][b]] * n_cands
                else:
                    n_cands = int(np.max(data_dict['seg'][b]))
                    clusters = data_dict['seg'][b]

                rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])  # separate clusters and concat
                for rix, r in enumerate(rois):
                    if np.sum(r !=0) > 0: #check if the lesion survived data augmentation
                        seg_ixs = np.argwhere(r != 0)
                        coord_list = [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                         np.max(seg_ixs[:, 2])+1]
                        if dim == 3:

                            coord_list.extend([np.min(seg_ixs[:, 3])-1, np.max(seg_ixs[:, 3])+1])

                        p_coords_list.append(coord_list)
                        p_roi_masks_list.append(r)
                        # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
                        # also patient wide, this assignment is not dependent on patch occurrances.
                        p_roi_labels_list.append(data_dict['class_target'][b][rix] + 1)

                    if class_specific_seg_flag:
                        out_seg[b][data_dict['seg'][b] == rix + 1] = data_dict['class_target'][b][rix] + 1

                if not class_specific_seg_flag:
                    out_seg[b][data_dict['seg'][b] > 0] = 1

                bb_target.append(np.array(p_coords_list))
                roi_masks.append(np.array(p_roi_masks_list).astype('uint8'))
                roi_labels.append(np.array(p_roi_labels_list))


            else:
                bb_target.append([])
                roi_masks.append(np.zeros_like(data_dict['seg'][b])[None])
                roi_labels.append(np.array([-1]))

        if get_rois_from_seg_flag:
            data_dict.pop('class_target', None)

        data_dict['bb_target'] = np.array(bb_target)
        data_dict['roi_masks'] = np.array(roi_masks)
        data_dict['class_target'] = np.array(roi_labels)
        data_dict['seg'] = out_seg

        return data_dict


def transpose_channels(batch):
    if len(batch.shape) == 4:
        return np.transpose(batch, axes=[0, 2, 3, 1])
    elif len(batch.shape) == 5:
        return np.transpose(batch, axes=[0, 4, 2, 3, 1])
    else:
        raise ValueError("wrong dimensions in transpose_channel generator!")


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def resize_multichannel_image(multichannel_image, new_shape, order=3):
    '''
    Resizes multichannel_image. Resizes each channel in c separately and fuses results back together

    :param multichannel_image: c x x x y (x z)
    :param new_shape: x x y (x z)
    :param order:
    :return:
    '''
    tpe = multichannel_image.dtype
    new_shp = [multichannel_image.shape[0]] + list(new_shape)
    result = np.zeros(new_shp, dtype=multichannel_image.dtype)
    for i in range(multichannel_image.shape[0]):
        result[i] = resize(multichannel_image[i].astype(float), new_shape, order, "constant", 0, True, anti_aliasing=False)
    return result.astype(tpe)


def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value


def uniform(low, high, size=None):
    """
    wrapper for np.random.uniform to allow it to handle low=high
    :param low:
    :param high:
    :return:
    """
    if low == high:
        if size is None:
            return low
        else:
            return np.ones(size) * low
    else:
        return np.random.uniform(low, high, size)


def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


def mask_random_square(img, square_size, n_val, channel_wise_n_val=False, square_pos=None):
    """Masks (sets = 0) a random square in an image"""

    img_h = img.shape[-2]
    img_w = img.shape[-1]

    img = img.copy()

    if square_pos is None:
        w_start = np.random.randint(0, img_w - square_size)
        h_start = np.random.randint(0, img_h - square_size)
    else:
        pos_wh = square_pos[np.random.randint(0, len(square_pos))]
        w_start = pos_wh[0]
        h_start = pos_wh[1]

    if img.ndim == 2:
        rnd_n_val = get_range_val(n_val)
        img[h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
    elif img.ndim == 3:
        if channel_wise_n_val:
            for i in range(img.shape[0]):
                rnd_n_val = get_range_val(n_val)
                img[i, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
        else:
            rnd_n_val = get_range_val(n_val)
            img[:, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
    elif img.ndim == 4:
        if channel_wise_n_val:
            for i in range(img.shape[0]):
                rnd_n_val = get_range_val(n_val)
                img[:, i, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val
        else:
            rnd_n_val = get_range_val(n_val)
            img[:, :, h_start:(h_start + square_size), w_start:(w_start + square_size)] = rnd_n_val

    return img


def mask_random_squares(img, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None):
    """Masks a given number of squares in an image"""
    for i in range(n_squares):
        img = mask_random_square(img, square_size, n_val, channel_wise_n_val=channel_wise_n_val,
                                 square_pos=square_pos)
    return img
