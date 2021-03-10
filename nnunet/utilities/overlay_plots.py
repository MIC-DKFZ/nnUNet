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
from multiprocessing.pool import Pool

import numpy as np
import SimpleITK as sitk

color_cycle = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
    "e6194B",
    "911eb4",
    "ffe119",
    "bfef45",
    "42d4f4",
    "f032e6",
    "000075",
    "9A6324",
    "808000",
    "800000",
    "469990",
)


def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def generate_overlay(input_image: np.ndarray, segmentation: np.ndarray, mapping: dict = None, color_cycle=color_cycle,
                     overlay_intensity=0.6):
    """
    image must be a color image, so last dimension must be 3. if image is grayscale, tile it first!
    Segmentation must be label map of same shape as image (w/o color channels)
    mapping can be label_id -> idx_in_cycle or None

    returned image is scaled to [0, 255]!!!
    """
    # assert len(image.shape) == len(segmentation.shape)
    # assert all([i == j for i, j in zip(image.shape, segmentation.shape)])

    # create a copy of image
    image = np.copy(input_image)

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif len(image.shape) == 3:
        assert image.shape[2] == 3, 'if 3d image is given the last dimension must be the color channels ' \
                                    '(3 channels). Only 2D images are supported'

    else:
        raise RuntimeError("unexpected image shape. only 2D images and 2D images with color channels (color in "
                           "last dimension) are supported")

    # rescale image to [0, 255]
    image = image - image.min()
    image = image / image.max() * 255

    # create output

    if mapping is None:
        uniques = np.unique(segmentation)
        mapping = {i: c for c, i in enumerate(uniques)}

    for l in mapping.keys():
        image[segmentation == l] += overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[l]]))

    # rescale result to [0, 255]
    image = image / image.max() * 255
    return image.astype(np.uint8)


def plot_overlay(image_file: str, segmentation_file: str, output_file: str, overlay_intensity: float = 0.6):
    import matplotlib.pyplot as plt

    image = sitk.GetArrayFromImage(sitk.ReadImage(image_file))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_file))
    assert all([i == j for i, j in zip(image.shape, seg.shape)]), "image and seg do not have the same shape: %s, %s" % (
        image_file, segmentation_file)

    assert len(image.shape) == 3, 'only 3D images/segs are supported'

    fg_mask = seg != 0
    fg_per_slice = fg_mask.sum((1, 2))
    selected_slice = np.argmax(fg_per_slice)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)

    plt.imsave(output_file, overlay)


def multiprocessing_plot_overlay(list_of_image_files, list_of_seg_files, list_of_output_files, overlay_intensity,
                                 num_processes=8):
    p = Pool(num_processes)
    r = p.starmap_async(plot_overlay, zip(
        list_of_image_files, list_of_seg_files, list_of_output_files, [overlay_intensity] * len(list_of_output_files)
    ))
    r.get()
    p.close()
    p.join()
