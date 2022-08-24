from typing import Union, Tuple

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


def get_ensemble_name(model1_folder, model2_folder, folds: Tuple[int, ...]):
    identifier = 'ensemble___' + os.path.basename(model1_folder) + '___' + \
                 os.path.basename(model2_folder) + '___' + folds_tuple_to_string(folds)
    return identifier


def get_ensemble_name_from_d_tr_c(dataset, tr1, p1, c1, tr2, p2, c2, folds: Tuple[int, ...]):
    model1_folder = get_output_folder(dataset, tr1, p1, c1)
    model2_folder = get_output_folder(dataset, tr2, p2, c2)

    get_ensemble_name(model1_folder, model2_folder, folds)


def convert_ensemble_folder_to_model_identifiers_and_folds(ensemble_folder: str):
    prefix, *models, folds = ensemble_folder.split('___')
    return models, folds


def folds_tuple_to_string(folds: Union[List[int], Tuple[int, ...]]):
    s = str(folds[0])
    for f in folds[1:]:
        s += f"_{f}"
    return s


def folds_string_to_tuple(folds_string: str):
    folds = folds_string.split('_')
    res = []
    for f in folds:
        try:
            res.append(int(f))
        except ValueError:
            res.append(f)
    return res


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
