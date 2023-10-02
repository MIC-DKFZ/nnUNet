import argparse
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, load_json, subfiles, save_json

from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.file_path_utilities import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


def move_plans_between_datasets(
        source_dataset_name_or_id: Union[int, str],
        target_dataset_name_or_id: Union[int, str],
        source_plans_identifier: str,
        target_plans_identifier: str = None):
    source_dataset_name = maybe_convert_to_dataset_name(source_dataset_name_or_id)
    target_dataset_name = maybe_convert_to_dataset_name(target_dataset_name_or_id)

    if target_plans_identifier is None:
        target_plans_identifier = source_plans_identifier

    source_folder = join(nnUNet_preprocessed, source_dataset_name)
    assert isdir(source_folder), f"Cannot move plans because preprocessed directory of source dataset is missing. " \
                                 f"Run nnUNetv2_plan_and_preprocess for source dataset first!"

    source_plans_file = join(source_folder, source_plans_identifier + '.json')
    assert isfile(source_plans_file), f"Source plans are missing. Run the corresponding experiment planning first! " \
                                      f"Expected file: {source_plans_file}"

    source_plans = load_json(source_plans_file)
    source_plans['dataset_name'] = target_dataset_name

    # we need to change data_identifier to use target_plans_identifier
    if target_plans_identifier != source_plans_identifier:
        for c in source_plans['configurations'].keys():
            if 'data_identifier' in source_plans['configurations'][c].keys():
                old_identifier = source_plans['configurations'][c]["data_identifier"]
                if old_identifier.startswith(source_plans_identifier):
                    new_identifier = target_plans_identifier + old_identifier[len(source_plans_identifier):]
                else:
                    new_identifier = target_plans_identifier + '_' + old_identifier
                source_plans['configurations'][c]["data_identifier"] = new_identifier

    # we need to change the reader writer class!
    target_raw_data_dir = join(nnUNet_raw, target_dataset_name)
    target_dataset_json = load_json(join(target_raw_data_dir, 'dataset.json'))

    # we may need to change the reader/writer
    # pick any file from the source dataset
    dataset = get_filenames_of_train_images_and_targets(target_raw_data_dir, target_dataset_json)
    example_image = dataset[dataset.keys().__iter__().__next__()]['images'][0]
    rw = determine_reader_writer_from_dataset_json(target_dataset_json, example_image, allow_nonmatching_filename=True,
                                                   verbose=False)

    source_plans["image_reader_writer"] = rw.__name__

    save_json(source_plans, join(nnUNet_preprocessed, target_dataset_name, target_plans_identifier + '.json'),
              sort_keys=False)


def entry_point_move_plans_between_datasets():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, required=True,
                        help='Source dataset name or id')
    parser.add_argument('-t', type=str, required=True,
                        help='Target dataset name or id')
    parser.add_argument('-sp', type=str, required=True,
                        help='Source plans identifier. If your plans are named "nnUNetPlans.json" then the '
                             'identifier would be nnUNetPlans')
    parser.add_argument('-tp', type=str, required=False, default=None,
                        help='Target plans identifier. Default is None meaning the source plans identifier will '
                             'be kept. Not recommended if the source plans identifier is a default nnU-Net identifier '
                             'such as nnUNetPlans!!!')
    args = parser.parse_args()
    move_plans_between_datasets(args.s, args.t, args.sp, args.tp)


if __name__ == '__main__':
    move_plans_between_datasets(2, 4, 'nnUNetPlans', 'nnUNetPlansFrom2')
