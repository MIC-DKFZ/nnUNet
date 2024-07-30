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
import multiprocessing
from multiprocessing.pool import Pool
from typing import Tuple, Union

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
import nnunetv2.paths as paths
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class, nnUNetBaseDataset
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    get_filenames_of_train_images_and_targets

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

    if image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif image.ndim == 3:
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
        uniques = np.sort(pd.unique(segmentation.ravel()))  # np.unique(segmentation)
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

    selects the slice with the largest amount of fg (how much percent of each class are in each slice? pick slice
    with highest avg percent)

    we give image so that we can easily replace this function if needed
    """
    classes = [i for i in np.sort(pd.unique(segmentation.ravel())) if i > 0]
    fg_per_slice = np.zeros((image.shape[0], len(classes)))
    for i, c in enumerate(classes):
        fg_mask = segmentation == c
        fg_per_slice[:, i] = fg_mask.sum((1, 2))
        fg_per_slice[:, i] /= fg_per_slice.sum()
    fg_per_slice = fg_per_slice.mean(1)
    return int(np.argmax(fg_per_slice))


def plot_overlay(image_file: str, segmentation_file: str, image_reader_writer: BaseReaderWriter, output_file: str,
                 overlay_intensity: float = 0.6):
    import matplotlib.pyplot as plt

    image, props = image_reader_writer.read_images((image_file, ))
    image = image[0]
    seg, props_seg = image_reader_writer.read_seg(segmentation_file)
    seg = seg[0]

    assert image.shape == seg.shape, "image and seg do not have the same shape: %s, %s" % (
        image_file, segmentation_file)

    assert image.ndim == 3, 'only 3D images/segs are supported'

    selected_slice = select_slice_to_plot2(image, seg)
    # print(image.shape, selected_slice)

    overlay = generate_overlay(image[selected_slice], seg[selected_slice], overlay_intensity=overlay_intensity)

    plt.imsave(output_file, overlay)


def plot_overlay_preprocessed(dataset: nnUNetBaseDataset, k: str, output_folder: str, overlay_intensity: float = 0.6, channel_idx=0):
    import matplotlib.pyplot as plt
    data, seg, _, properties = dataset.load_case(k)

    assert channel_idx < (data.shape[0]), 'This dataset only supports channel index up to %d' % (data.shape[0] - 1)

    image = data[channel_idx]
    seg = seg[0]
    selected_slice = select_slice_to_plot2(image, seg)

    seg = np.copy(seg[selected_slice])
    seg[seg < 0] = 0
    overlay = generate_overlay(image[selected_slice], seg, overlay_intensity=overlay_intensity)

    plt.imsave(join(output_folder, k + '.png'), overlay)


def multiprocessing_plot_overlay(list_of_image_files, list_of_seg_files, image_reader_writer,
                                 list_of_output_files, overlay_intensity,
                                 num_processes=8):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = p.starmap_async(plot_overlay, zip(
            list_of_image_files, list_of_seg_files, [image_reader_writer] * len(list_of_output_files),
            list_of_output_files, [overlay_intensity] * len(list_of_output_files)
        ))
        r.get()


def multiprocessing_plot_overlay_preprocessed(dataset: nnUNetBaseDataset, output_folder, overlay_intensity,
                                              num_processes=8, channel_idx=0):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        r = []
        for k in dataset.identifiers:
            r.append(
                p.starmap_async(plot_overlay_preprocessed,
                                ((
                                    dataset, k, output_folder, overlay_intensity, channel_idx
                                 ),))
            )
        _ = [i.get() for i in r]


def generate_overlays_from_raw(dataset_name_or_id: Union[int, str], output_folder: str,
                               num_processes: int = 8, channel_idx: int = 0, overlay_intensity: float = 0.6):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = join(paths.nnUNet_raw, dataset_name)
    dataset_json = load_json(join(folder, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(folder, dataset_json)

    image_files = [v['images'][channel_idx] for v in dataset.values()]
    seg_files = [v['label'] for v in dataset.values()]

    assert all([isfile(i) for i in image_files])
    assert all([isfile(i) for i in seg_files])

    maybe_mkdir_p(output_folder)
    output_files = [join(output_folder, i + '.png') for i in dataset.keys()]

    image_reader_writer = determine_reader_writer_from_dataset_json(dataset_json, image_files[0])()
    multiprocessing_plot_overlay(image_files, seg_files, image_reader_writer, output_files, overlay_intensity, num_processes)


def generate_overlays_from_preprocessed(dataset_name_or_id: Union[int, str], output_folder: str,
                                        num_processes: int = 8, channel_idx: int = 0,
                                        configuration: str = None,
                                        plans_identifier: str = 'nnUNetPlans',
                                        overlay_intensity: float = 0.6):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = join(paths.nnUNet_preprocessed, dataset_name)
    if not isdir(folder): raise RuntimeError("run preprocessing for that task first")

    plans = load_json(join(folder, plans_identifier + '.json'))
    if configuration is None:
        if '3d_fullres' in plans['configurations'].keys():
            configuration = '3d_fullres'
        else:
            configuration = '2d'
    cm = ConfigurationManager(plans['configurations'][configuration])
    preprocessed_folder = join(folder, cm.data_identifier)

    if not isdir(preprocessed_folder):
        raise RuntimeError(f"Preprocessed data folder for configuration {configuration} of plans identifier "
                           f"{plans_identifier} ({dataset_name}) does not exist. Run preprocessing for this "
                           f"configuration first!")

    dc = infer_dataset_class(preprocessed_folder)
    dataset = dc(preprocessed_folder)

    maybe_mkdir_p(output_folder)
    multiprocessing_plot_overlay_preprocessed(dataset, output_folder, overlay_intensity=overlay_intensity,
                                              num_processes=num_processes, channel_idx=channel_idx)


def entry_point_generate_overlay():
    import argparse
    parser = argparse.ArgumentParser("Plots png overlays of the slice with the most foreground. Note that this "
                                     "disregards spacing information!")
    parser.add_argument('-d', type=str, help="Dataset name or id", required=True)
    parser.add_argument('-o', type=str, help="output folder", required=True)
    parser.add_argument('-np', type=int, default=default_num_processes, required=False,
                        help=f"number of processes used. Default: {default_num_processes}")
    parser.add_argument('-channel_idx', type=int, default=0, required=False,
                        help="channel index used (0 = _0000). Default: 0")
    parser.add_argument('--use_raw', action='store_true', required=False, help="if set then we use raw data. else "
                                                                               "we use preprocessed")
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='plans identifier. Only used if --use_raw is not set! Default: nnUNetPlans')
    parser.add_argument('-c', type=str, required=False, default=None,
                        help='configuration name. Only used if --use_raw is not set! Default: None = '
                             '3d_fullres if available, else 2d')
    parser.add_argument('-overlay_intensity', type=float, required=False, default=0.6,
                        help='overlay intensity. Higher = brighter/less transparent')


    args = parser.parse_args()

    if args.use_raw:
        generate_overlays_from_raw(args.d, args.o, args.np, args.channel_idx,
                                   overlay_intensity=args.overlay_intensity)
    else:
        generate_overlays_from_preprocessed(args.d, args.o, args.np, args.channel_idx, args.c, args.p,
                                            overlay_intensity=args.overlay_intensity)


if __name__ == '__main__':
    entry_point_generate_overlay()
