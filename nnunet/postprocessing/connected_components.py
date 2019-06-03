#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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
from scipy.ndimage import label
import SimpleITK as sitk
from nnunet.utilities.sitk_stuff import copy_geometry


def load_remove_save(input_file: str, output_file: str, for_which_classes: list):
    img_in = sitk.ReadImage(input_file)
    img_npy = sitk.GetArrayFromImage(img_in)

    img_out_itk = sitk.GetImageFromArray(remove_all_but_the_largest_connected_component(img_npy, for_which_classes))
    img_out_itk = copy_geometry(img_out_itk, img_in)
    sitk.WriteImage(img_out_itk, output_file)


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes

    for c in for_which_classes:
        mask = image == c
        lmap, num_objects = label(mask.astype(int))
        if num_objects > 1:
            sizes = []
            for o in range(1, num_objects + 1):
                sizes.append((lmap == o).sum())
            mx = np.argmax(sizes) + 1
            image[(lmap != mx) & mask] = 0
    return image
