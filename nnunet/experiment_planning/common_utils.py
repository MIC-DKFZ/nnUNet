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

import numpy as np
from copy import deepcopy
from nnunet.network_architecture.generic_UNet import Generic_UNet
import SimpleITK as sitk
import shutil
from batchgenerators.utilities.file_and_folder_operations import join


def split_4d_nifti(filename, output_folder):
    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = filename.split("/")[-1]
    if dim == 3:
        shutil.copy(filename, join(output_folder, file_base[:-7] + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, filename))
    else:
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection()).reshape(4,4)
        # now modify these to remove the fourth dimension
        spacing = tuple(list(spacing[:-1]))
        origin = tuple(list(origin[:-1]))
        direction = tuple(direction[:-1, :-1].reshape(-1))
        for i, t in enumerate(range(img_npy.shape[0])):
            img = img_npy[t]
            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(spacing)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(img_itk_new, join(output_folder, file_base[:-7] + "_%04.0d.nii.gz" % i))


def get_pool_and_conv_props_poolLateV2(patch_size, min_feature_map_size, max_numpool, spacing):
    """

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :return:
    """
    initial_spacing = deepcopy(spacing)
    reach = max(initial_spacing)
    dim = len(patch_size)

    num_pool_per_axis = get_network_numpool(patch_size, max_numpool, min_feature_map_size)

    net_num_pool_op_kernel_sizes = []
    net_conv_kernel_sizes = []
    net_numpool = max(num_pool_per_axis)

    current_spacing = spacing
    for p in range(net_numpool):
        reached = [current_spacing[i] / reach > 0.5 for i in range(dim)]
        pool = [2 if num_pool_per_axis[i] + p >= net_numpool else 1 for i in range(dim)]
        if all(reached):
            conv = [3] * dim
        else:
            conv = [3 if not reached[i] else 1 for i in range(dim)]
        net_num_pool_op_kernel_sizes.append(pool)
        net_conv_kernel_sizes.append(conv)
        current_spacing = [i * j for i, j in zip(current_spacing, pool)]

    net_conv_kernel_sizes.append([3] * dim)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    return num_pool_per_axis, net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, patch_size, must_be_divisible_by


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :return:
    """
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = []
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim

    while True:
        # find axes that are within factor 2 of min axis spacing
        # TODO this is a problem because sometimes we have spacing 20, 50, 50 and we want to still keep pooling.
        #  Here we would stop however. This is not what we want! We need to restrict min_spacing = min(current_spacing)
        #  to only the spacings that belong to axes that can still be pooled :-/
        min_spacing = min(current_spacing)
        valid_axes_for_pool = [i for i in range(dim) if current_spacing[i] / min_spacing < 2]
        axes = []
        for a in range(dim):
            my_spacing = current_spacing[a]
            partners = [i for i in range(dim) if current_spacing[i] / my_spacing < 2 and my_spacing / current_spacing[i] < 2]
            if len(partners) > len(axes):
                axes = partners
        conv_kernel_size = [3 if i in axes else 1 for i in range(dim)]

        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        #before = len(valid_axes_for_pool)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_size[i] >= 2*min_feature_map_size]
        #after = len(valid_axes_for_pool)
        #if after == 1 and before > 1:
        #    break

        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 0:
            break

        #print(current_spacing, current_size)

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] /= 2
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(conv_kernel_size)
        #print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3]*dim)
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by


def get_shape_must_be_divisible_by(net_numpool_per_axis):
    return 2 ** np.array(net_numpool_per_axis)


def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisibly by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp


def get_network_numpool(patch_size, maxpool_cap=999, min_feature_map_size=4):
    network_numpool_per_axis = np.floor([np.log(i / min_feature_map_size) / np.log(2) for i in patch_size]).astype(int)
    network_numpool_per_axis = [min(i, maxpool_cap) for i in network_numpool_per_axis]
    return network_numpool_per_axis
