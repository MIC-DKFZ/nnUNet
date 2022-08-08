from typing import Union

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def convert_trainer_plans_config_to_identifier(trainer_name, plans_identifier, configuration):
    return f'{trainer_name}__{plans_identifier}__{configuration}'


def convert_identifier_to_trainer_plans_config(identifier: str):
    return identifier.split('__')


def get_output_folder(dataset_name_or_id: Union[str, int], trainer_name: str = 'nnUNetTrainer',
                      plans_identifier: str = 'nnUNetPlans', configuration: str = '3d_fullres',
                      fold: Union[str, int] = None) -> str:
    tmp = join(nnUNet_results, maybe_convert_to_dataset_name(dataset_name_or_id),
               convert_trainer_plans_config_to_identifier(trainer_name, plans_identifier, configuration))
    if fold is not None:
        tmp = join(tmp, f'fold_{fold}')
    return tmp


def parse_dataset_trainer_plans_configuration_from_path(path: str):
    folders = split_path(path)
    # this here can be a little tricky because we are making assumptions. Let's hope this never fails lol

    # safer to make this depend on two conditions, the fold_x and the DatasetXXX
    # first let's see if some fold_X is present
    fold_x_present = [i.startswith('fold_') for i in folders]
    if any(fold_x_present):
        idx = fold_x_present.index(True)
        # OK now two entries before that there should be DatasetXXX
        assert len(folders[:idx]) >= 2, 'Bad path, cannot extract what I need. Your path needs to be at least ' \
                                        'DatasetXXX/MODULE__PLANS__CONFIGURATION for this to work'
        if folders[idx - 2].startswith('Dataset'):
            splitted = folders[idx - 1].split('__')
            assert len(splitted) == 3, 'Bad path, cannot extract what I need. Your path needs to be at least ' \
                                        'DatasetXXX/MODULE__PLANS__CONFIGURATION for this to work'
            return folders[idx - 2], *splitted
    else:
        # we can only check for dataset followed by a string that is separable into three strings by splitting with '__'
        # look for DatasetXXX
        dataset_folder = [i.startswith('Dataset') for i in folders]
        if any(dataset_folder):
            idx = dataset_folder.index(True)
            assert len(folders) >= (idx + 1), 'Bad path, cannot extract what I need. Your path needs to be at least ' \
                                        'DatasetXXX/MODULE__PLANS__CONFIGURATION for this to work'
            splitted = folders[idx + 1].split('__')
            assert len(splitted) == 3, 'Bad path, cannot extract what I need. Your path needs to be at least ' \
                                       'DatasetXXX/MODULE__PLANS__CONFIGURATION for this to work'
            return folders[idx], *splitted


if __name__ == '__main__':
    ### well at this point I could just write tests...
    path = '/home/fabian/results/nnUNet_remake/Dataset002_Heart/nnUNetModule__nnUNetPlans__3d_fullres'
    print(parse_dataset_trainer_plans_configuration_from_path(path))
    path = 'Dataset002_Heart/nnUNetModule__nnUNetPlans__3d_fullres'
    print(parse_dataset_trainer_plans_configuration_from_path(path))
    path = '/home/fabian/results/nnUNet_remake/Dataset002_Heart/nnUNetModule__nnUNetPlans__3d_fullres/fold_all'
    print(parse_dataset_trainer_plans_configuration_from_path(path))
    try:
        path = '/home/fabian/results/nnUNet_remake/Dataset002_Heart/'
        print(parse_dataset_trainer_plans_configuration_from_path(path))
    except AssertionError:
        print('yayy, assertion works')
