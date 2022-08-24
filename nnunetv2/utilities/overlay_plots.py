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
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *

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


def generate_overlay(input_image: np.ndarray, segmentation: np.ndarray, mapping: dict = None,
                     color_cycle: Tuple[str, ...] = color_cycle,
                     overlay_intensity: float = 0.6):
    """
    image can be 2d greyscale or 2d RGB (color channel in last dimension!)

    Segmentation must be label map of same shape as image (w/o color channels)

    mapping can be label_id -> idx_in_cycle or None

    returned image is scaled to [0, 255] (uint8)!!!
    """
    # create a copy of image
    image = np.copy(input_image)

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            image = np.tile(image, (1, 1, 3))
        else:
            raise RuntimeError(f'if 3d image is given the last dimension must be the color channels (3 channels). '
                               f'Only 2D images are supported. Your image shape: {image.shape}')
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


def select_slice_to_plot(image: np.ndarray, segmentation: np.ndarray) -> int:
    """
    image and segmentation are expected to be 3D

    selects the slice with the largest amount of fg (regardless of label)

    we give image so that we can easily replace this function if needed
    """
    fg_mask = segmentation != 0
    fg_per_slice = fg_mask.sum((1, 2))
    selected_slice = int(np.argmax(fg_per_slice))
    return selected_slice


def select_slice_to_plot2(image: np.ndarray, segmentation: np.ndarray) -> int:
    """
    image and segmentation are expected to be 3D (or 1, x, y)

    selects the slice with the largest amount of fg (average fg per class)

    we give image so that we can easily replace this function if needed
    """
    classes = [i for i in np.unique(segmentation) if i != 0]
    fg_per_slice = []
    for c in classes:
        fg_mask = segmentation == c
        fg_per_slice = fg_mask.sum((1, 2))


def plot_overlay(image_file: str, segmentation_file: str, output_file: str, overlay_intensity: float = 0.6):
    import matplotlib.pyplot as plt

    image = sitk.GetArrayFromImage(sitk.ReadImage(image_file))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_file))
    assert all([i == j for i, j in zip(image.shape, seg.shape)]), "image and seg do not have the same shape: %s, %s" % (
        image_file, segmentation_file)

    assert len(image.shape) == 3, 'only 3D images/segs are supported'

    selected_slice = select_slice_to_plot(image, seg)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)

    plt.imsave(output_file, overlay)


def plot_overlay_preprocessed(case_file: str, output_file: str, overlay_intensity: float = 0.6, modality_index=0):
    import matplotlib.pyplot as plt
    data = np.load(case_file)['data']

    assert modality_index < (data.shape[0] - 1), 'This dataset only supports modality index up to %d' % (data.shape[0] - 2)

    image = data[modality_index]
    seg = data[-1]
    seg[seg < 0] = 0

    fg_mask = seg > 0
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


def multiprocessing_plot_overlay_preprocessed(list_of_case_files, list_of_output_files, overlay_intensity,
                                 num_processes=8, modality_index=0):
    p = Pool(num_processes)
    r = p.starmap_async(plot_overlay_preprocessed, zip(
        list_of_case_files, list_of_output_files, [overlay_intensity] * len(list_of_output_files),
        [modality_index] * len(list_of_output_files)
    ))
    r.get()
    p.close()
    p.join()


def generate_overlays_for_task(task_name_or_id, output_folder, num_processes=8, modality_idx=0, use_preprocessed=True,
                               data_identifier=default_data_identifier):
    if isinstance(task_name_or_id, str):
        if not task_name_or_id.startswith("Task"):
            task_name_or_id = int(task_name_or_id)
            task_name = convert_id_to_task_name(task_name_or_id)
        else:
            task_name = task_name_or_id
    else:
        task_name = convert_id_to_task_name(int(task_name_or_id))

    if not use_preprocessed:
        folder = join(nnUNet_raw_data, task_name)

        identifiers = [i[:-7] for i in subfiles(join(folder, 'labelsTr'), suffix='.nii.gz', join=False)]

        image_files = [join(folder, 'imagesTr', i + "_%04.0d.nii.gz" % modality_idx) for i in identifiers]
        seg_files = [join(folder, 'labelsTr', i + ".nii.gz") for i in identifiers]

        assert all([isfile(i) for i in image_files])
        assert all([isfile(i) for i in seg_files])

        maybe_mkdir_p(output_folder)
        output_files = [join(output_folder, i + '.png') for i in identifiers]
        multiprocessing_plot_overlay(image_files, seg_files, output_files, 0.6, num_processes)
    else:
        folder = join(preprocessing_output_dir, task_name)
        if not isdir(folder): raise RuntimeError("run preprocessing for that task first")
        matching_folders = subdirs(folder, prefix=data_identifier + "_stage")
        if len(matching_folders) == 0: "run preprocessing for that task first (use default experiment planner!)"
        matching_folders.sort()
        folder = matching_folders[-1]
        identifiers = [i[:-4] for i in subfiles(folder, suffix='.npz', join=False)]
        maybe_mkdir_p(output_folder)
        output_files = [join(output_folder, i + '.png') for i in identifiers]
        image_files = [join(folder, i + ".npz") for i in identifiers]
        maybe_mkdir_p(output_folder)
        multiprocessing_plot_overlay_preprocessed(image_files, output_files, overlay_intensity=0.6,
                                                  num_processes=num_processes, modality_index=modality_idx)


def entry_point_generate_overlay():
    import argparse
    parser = argparse.ArgumentParser("Plots png overlays of the slice with the most foreground. Note that this "
                                     "disregards spacing information!")
    parser.add_argument('-t', type=str, help="task name or task ID", required=True)
    parser.add_argument('-o', type=str, help="output folder", required=True)
    parser.add_argument('-num_processes', type=int, default=8, required=False, help="number of processes used. Default: 8")
    parser.add_argument('-modality_idx', type=int, default=0, required=False,
                        help="modality index used (0 = _0000.nii.gz). Default: 0")
    parser.add_argument('--use_raw', action='store_true', required=False, help="if set then we use raw data. else "
                                                                               "we use preprocessed")
    args = parser.parse_args()

    generate_overlays_for_task(args.t, args.o, args.num_processes, args.modality_idx, use_preprocessed=not args.use_raw)