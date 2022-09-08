from typing import Union

from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, load_json, subfiles, save_json

from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.file_path_utilities import maybe_convert_to_dataset_name


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
            old_identifier = source_plans['configurations'][c]["data_identifier"]
            if old_identifier.startswith(source_plans_identifier):
                new_identifier = target_plans_identifier + old_identifier[len(source_plans_identifier):]
            else:
                new_identifier = target_plans_identifier + '_' + old_identifier
            source_plans['configurations'][c]["data_identifier"] = new_identifier

    # we need to change the reader writer class!
    target_raw_data_dir = join(nnUNet_raw, target_dataset_name)
    target_dataset_json = load_json(join(target_raw_data_dir, 'dataset.json'))
    file_ending = target_dataset_json['file_ending']
    # pick any file from the imagesTr folder
    some_file = subfiles(join(target_raw_data_dir, 'imagesTr'), suffix=file_ending)[0]
    rw = determine_reader_writer_from_dataset_json(target_dataset_json, some_file, allow_nonmatching_filename=True)

    source_plans["image_reader_writer"] = rw.__name__

    save_json(source_plans, join(nnUNet_preprocessed, target_dataset_name, target_plans_identifier + '.json'),
              sort_keys=False)


if __name__ == '__main__':
    move_plans_between_datasets(2, 4, 'nnUNetPlans', 'nnUNetPlansFrom2')
