from typing import Union

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def get_preprocessor_name_from_plans(plans: dict, configuration_name: str):
    if configuration_name not in plans['configurations'].keys():
        raise RuntimeError(f'Missing configuration {configuration_name} in plans')
    return plans['configurations'][configuration_name]["preprocessor_name"]


def get_preprocessor_class_from_plans(plans: dict, configuration_name: str):
    preprocessor_name = get_preprocessor_name_from_plans(plans, configuration_name)
    preprocessor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing"),
                                                     preprocessor_name,
                                                     current_module="nnunetv2.preprocessing")
    if preprocessor_class is None:
        raise RuntimeError(f'Could not find preprocessor class {preprocessor_name} in nnunetv2.preprocessing. If it '
                           f'is located somewhere else, please move it there.')
    return preprocessor_class


def get_preprocessor_name_from_plans_file(plans_file: str, configuration: str):
    return get_preprocessor_name_from_plans(load_json(plans_file), configuration)


def get_preprocessor_name_from_plans_identifier(dataset_name_or_id: Union[int, str], plans_identifier: str,
                                                configuration: str):
    plans_file = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id),
                      plans_identifier + '.json')
    return get_preprocessor_name_from_plans_file(plans_file, configuration)


def get_preprocessor_class_from_plans_identifier(dataset_name_or_id: Union[int, str], plans_identifier: str,
                                                 configuration: str):
    preprocessor_name = get_preprocessor_name_from_plans_identifier(dataset_name_or_id, plans_identifier, configuration)
    preprocessor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing"),
                                                     preprocessor_name,
                                                     current_module="nnunetv2.preprocessing")
    if preprocessor_class is None:
        raise RuntimeError(f'Could not find preprocessor class {preprocessor_name} in nnunetv2.preprocessing. If it '
                           f'is located somewhere else, please move it there.')
    return preprocessor_class


