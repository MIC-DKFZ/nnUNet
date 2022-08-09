import os.path
import shutil

from nnunetv2.configuration import default_num_processes
from typing import Union, List, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_json, join, isdir, maybe_mkdir_p, subfiles, isfile

from nnunetv2.ensembling.ensemble import ensemble_crossvalidations
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, load_summary_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.utilities.file_path_utilities import maybe_convert_to_dataset_name, get_output_folder, \
    convert_identifier_to_trainer_plans_config
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager

default_trained_models = tuple([
    {'plans': 'nnUNetPlans', 'configuration': '2d', 'trainer': 'nnUNetTrainer'},
    {'plans': 'nnUNetPlans', 'configuration': '3d_fullres', 'trainer': 'nnUNetTrainer'},
    {'plans': 'nnUNetPlans', 'configuration': '3d_lowres', 'trainer': 'nnUNetTrainer'},
    {'plans': 'nnUNetPlans', 'configuration': '3d_cascade_fullres', 'trainer': 'nnUNetTrainer'},
])


def filter_available_models(model_dict: dict, dataset_name_or_id: Union[str, int]):
    valid = []
    for trained_model in model_dict:
        plans = load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id),
                               trained_model['plans'] + '.json'))
        # check if configuration exists
        # 3d_cascade_fullres and 3d_lowres do not exist for each dataset so we allow them to be absent IF they are not
        # specified in the plans file
        if trained_model['configuration'] not in plans['configurations'].keys():
            print(f"Configuration {trained_model['configuration']} not found in plans {trained_model['plans']}.\n"
                  f"Inferred plans file: {join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id), trained_model['plans'] + '.json')}.")
            continue

        # check if trained model output folder exists. This is a requirement. No mercy here.
        expected_output_folder = get_output_folder(dataset_name_or_id, trained_model['trainer'], trained_model['plans'],
                                                   trained_model['configuration'], fold=None)
        if not isdir(expected_output_folder):
            raise RuntimeError(f"Trained model {trained_model} does not have an output folder. "
                  f"Expected: {expected_output_folder}. Please run the training for this model! (don't forget "
                  f"the --npz flag if you want to ensemble multiple configurations)")

        valid.append(trained_model)
    return valid


def folds_tuple_to_string(folds: Union[List[int], Tuple[int, ...]]):
    s = str(folds[0])
    for f in folds[1:]:
        s += f"_{f}"
    return s


def accumulate_cv_results(trained_model_folder,
                          merged_output_folder: str,
                          folds: Union[List[int], Tuple[int, ...]],
                          num_processes: int = default_num_processes,
                          overwrite: bool = True):
    """
    There are a lot of things that can get fucked up, so the simplest way to deal with potential problems is to
    collect the cv results into a separate folder and then evaluate them again. No messing with summary_json files!
    """

    if overwrite and isdir(merged_output_folder):
        shutil.rmtree(merged_output_folder)
    maybe_mkdir_p(merged_output_folder)

    dataset_json = load_json(join(trained_model_folder, 'dataset.json'))
    plans = load_json(join(trained_model_folder, 'plans.json'))

    did_we_copy_something = False
    for f in folds:
        expected_validation_folder = join(trained_model_folder, f'fold_{f}', 'validation')
        predicted_files = subfiles(expected_validation_folder, suffix=dataset_json['file_ending'], join=False)
        for pf in predicted_files:
            if overwrite and isfile(join(merged_output_folder, pf)):
                raise RuntimeError(f'More than one of your folds has a prediction for case {pf}')
            if overwrite or not isfile(join(merged_output_folder, pf)):
                shutil.copy(join(expected_validation_folder, pf), join(merged_output_folder, pf))
                did_we_copy_something = True

    if did_we_copy_something or not isfile(join(merged_output_folder, 'summary.json')):
        label_manager = get_labelmanager(plans, dataset_json)
        compute_metrics_on_folder(join(nnUNet_raw, 'labelsTr'), merged_output_folder, join(merged_output_folder, 'summary.json'),
                                  plans['image_reader_writer'], dataset_json['file_ending'],
                                  label_manager.foreground_regions if label_manager.has_regions else
                                  label_manager.foreground_labels, label_manager.ignore_label, num_processes)


def generate_inference_command(dataset_name_or_id: Union[int, str], configuration_name: str,
                               plans_identifier: str = 'nnUNetPlans', trainer_name: str = 'nnUNetTrainer',
                               folds: Union[List[int], Tuple[int, ...]] = (0, 1, 2, 3, 4),
                               folder_with_segs_from_prev_stage: str = None,
                               input_folder: str = 'INPUT_FOLDER',
                               output_folder: str = 'OUTPUT_FOLDER',
                               save_npz: bool = False):
    fold_str = ''
    for f in folds:
        fold_str += f' {f}'

    predict_command = ''
    trained_model_folder = get_output_folder(dataset_name_or_id, trainer_name, plans_identifier, configuration_name, fold=None)
    plans = load_json(join(trained_model_folder, 'plans.json'))
    if 'previous_stage' in plans['configurations'][configuration_name].keys():
        prev_stage = plans['configurations'][configuration_name]['previous_stage']
        predict_command += generate_inference_command(dataset_name_or_id, prev_stage, plans_identifier, trainer_name,
                                                      folds, None, output_folder='OUTPUT_FOLDER_PREV_STAGE') + '\n'
        folder_with_segs_from_prev_stage = 'OUTPUT_FOLDER_PREV_STAGE'

    predict_command = f'nnUNetv2_predict -d {dataset_name_or_id} -i {input_folder} -o {output_folder} -f {fold_str} ' \
                      f'-tr {trainer_name} -c {configuration_name} -p {plans_identifier}'
    if folder_with_segs_from_prev_stage is not None:
        predict_command += f' -prev_stage_predictions {folder_with_segs_from_prev_stage}'
    if save_npz:
        predict_command += ' --npz'
    return predict_command


def find_best_configuration(dataset_name_or_id,
                            allowed_trained_models: List[dict] = default_trained_models,
                            allow_ensembling: bool = True,
                            num_processes: int = default_num_processes,
                            overwrite: bool = True,
                            folds: Union[List[int], Tuple[int, ...]] = (0, 1, 2, 3, 4)):
    all_results = {}
    for m in allowed_trained_models:
        output_folder = get_output_folder(dataset_name_or_id, m['trainer'], m['plans'], m['configuration'], fold=None)
        identifier = os.path.basename(output_folder)
        merged_output_folder = join(output_folder, f'crossval_results_folds_{folds_tuple_to_string(folds)}')
        accumulate_cv_results(output_folder, merged_output_folder, folds, num_processes, overwrite)
        all_results[identifier] = load_summary_json(join(merged_output_folder, 'summary.json'))['foreground_mean']['Dice']

    if allow_ensembling:
        for i in range(len(allowed_trained_models)):
            for j in range(i + 1, len(allowed_trained_models)):
                m1, m2 = allowed_trained_models[i], allowed_trained_models[j]
                output_folder_1 = get_output_folder(dataset_name_or_id, m1['trainer'], m1['plans'], m1['configuration'], fold=None)
                output_folder_2 = get_output_folder(dataset_name_or_id, m2['trainer'], m2['plans'], m2['configuration'], fold=None)
                identifier = 'ensemble___' + os.path.basename(output_folder_1) + '___' + \
                              os.path.basename(output_folder_2) + '___' + folds_tuple_to_string(folds)
                output_folder_ensemble = join(nnUNet_results, maybe_convert_to_dataset_name(dataset_name_or_id),
                                              'ensembles', identifier)
                ensemble_crossvalidations([output_folder_1, output_folder_2], output_folder_ensemble, folds, num_processes)
                all_results[identifier] = load_summary_json(join(output_folder_ensemble, 'summary.json'))['foreground_mean']['Dice']

    # pick best and report inference command
    best_score = max(all_results.values())
    best_keys = [k for k in all_results.keys() if all_results[k] == best_score]  # may never happen but theoretically
    # there can be a tie. Let's pick the first model in this case because it's going to be the simpler one (ensembles
    # come after single configs)
    best_key = best_keys[0]

    print('All results:')
    print(all_results)
    print(f'\nBest: {best_key}: {all_results[best_key]}')

    # convert best key to inference command:
    print('\nUse this for inference:')
    if best_key.startswith('ensemble___'):
        print('An ensemble won! What a surprise!')
        prefix, m1, m2 = best_key.split('___')
        tr1, pl1, c1 = convert_identifier_to_trainer_plans_config(m1)
        tr2, pl2, c2 = convert_identifier_to_trainer_plans_config(m1)
        print(generate_inference_command(dataset_name_or_id, c1, pl1, tr1, folds, save_npz=True, output_folder='OUTPUT_FOLDER_MODEL_1'))
        print(generate_inference_command(dataset_name_or_id, c2, pl2, tr2, folds, save_npz=True, output_folder='OUTPUT_FOLDER_MODEL_2'))
        raise RuntimeError('nnUNetv2_ensemble not yet implemented. Fabian wtf!?')
    else:
        tr, pl, c = convert_identifier_to_trainer_plans_config(best_key)
        print(generate_inference_command(dataset_name_or_id, c, pl, tr, folds))