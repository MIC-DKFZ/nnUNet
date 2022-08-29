from multiprocessing import Pool
from typing import Union, Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.configuration import default_num_processes
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def convert_trainer_plans_config_to_identifier(trainer_name, plans_identifier, configuration):
    return f'{trainer_name}__{plans_identifier}__{configuration}'


def convert_identifier_to_trainer_plans_config(identifier: str):
    return os.path.basename(identifier).split('__')


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
    prefix, *models, folds = os.path.basename(ensemble_folder).split('___')
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


def should_i_save_to_file(results_list: Union[None, List],
                          export_pool: Union[None, Pool],
                          prediction: np.ndarray):
    """
    There is a problem with python process communication that prevents us from communicating objects
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code. We circumvent that problem here by saving the data to a npy file that will
    then be read (and finally deleted) by the background Process. The code running in the background process must be
    implemented such that it can take either filename (str) or np.ndarray as input

    This function determines whether the object that should be passed through a multiprocessing pipe is too big.

    It also determines whether the export pool can keep up with its tasks and if not it will trigger
    saving results to disk in order to reduce the amoutn of RAM that is consumed (queued tasks can use a lot of RAM)

    We also check for dead workers and crash in case there are any. This should fix some peoples issues where
    the inference was just stuck (due to out of memory problems).

    Returns: True if we should save to file else False
    """
    if prediction.dtype in (np.float32, np.int32, np.uint32):
        bytes_per_element = 4
    elif prediction.dtype in (np.float, np.int, int, float, np.uint64):
        bytes_per_element = 8
    elif prediction.dtype in (np.uint8, np.int8, np.char):
        bytes_per_element = 2
    elif prediction == np.bool:
        bytes_per_element = 1  # is that so? I don't know tbh but like this its not going to crash for sure.
    else:
        raise RuntimeError(f'Unexpected dtype {prediction.dtype}')

    prediction_shape = prediction.shape
    if np.prod(prediction_shape) > (2e9 / bytes_per_element * 0.85):  # *0.85 just to be safe
        print('INFO: Prediction is too large for python process-process communication. Saving to file...')
        return True
    if export_pool is not None:
        is_alive = [i.is_alive for i in export_pool._pool]
        if not all(is_alive):
            raise RuntimeError("Some workers in the export pool are no longer alive. That should not happen. You "
                               "probably don't have enough RAM :-(")
        if results_list is not None:
            # We should prevent the task queue from getting too long. This could cause lots of predictions being
            # stuck in a queue and eating up memory. Best to save to disk instead in that case. Hopefully there
            # will be fewer people with RAM issues in the future...
            not_ready = [not i.ready() for i in results_list]
            if sum(not_ready) > len(is_alive):
                print('INFO: Prediction is faster than your PC can resample the results. Results are temporarily '
                      'saved to disk to prevent out of memory issues. If you have more RAM and CPU cores available, '
                      f'consider setting nnUNet_def_n_proc to a larger number (default is 8, current is '
                      f'{default_num_processes}).')
                return True
    return False


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
