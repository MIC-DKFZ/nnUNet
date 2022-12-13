import argparse
import os.path
import shutil
from copy import deepcopy
from typing import Union, List, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_json, join, isdir, maybe_mkdir_p, subfiles, isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.ensembling.ensemble import ensemble_crossvalidations
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, load_summary_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.utilities.file_path_utilities import maybe_convert_to_dataset_name, get_output_folder, \
    convert_identifier_to_trainer_plans_config, get_ensemble_name, folds_tuple_to_string
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

default_trained_models = tuple([
    {'plans': 'nnUNetPlans', 'configuration': '2d', 'trainer': 'nnUNetTrainer'},
    {'plans': 'nnUNetPlans', 'configuration': '3d_fullres', 'trainer': 'nnUNetTrainer'},
    {'plans': 'nnUNetPlans', 'configuration': '3d_lowres', 'trainer': 'nnUNetTrainer'},
    {'plans': 'nnUNetPlans', 'configuration': '3d_cascade_fullres', 'trainer': 'nnUNetTrainer'},
])


def filter_available_models(model_dict: Union[List[dict], Tuple[dict, ...]], dataset_name_or_id: Union[str, int]):
    valid = []
    for trained_model in model_dict:
        plans_manager = PlansManager(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id),
                               trained_model['plans'] + '.json'))
        # check if configuration exists
        # 3d_cascade_fullres and 3d_lowres do not exist for each dataset so we allow them to be absent IF they are not
        # specified in the plans file
        if trained_model['configuration'] not in plans_manager.available_configurations:
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
    plans_manager = PlansManager(join(trained_model_folder, 'plans.json'))
    rw = plans_manager.image_reader_writer_class()
    shutil.copy(join(trained_model_folder, 'dataset.json'), join(merged_output_folder, 'dataset.json'))
    shutil.copy(join(trained_model_folder, 'plans.json'), join(merged_output_folder, 'plans.json'))

    did_we_copy_something = False
    for f in folds:
        expected_validation_folder = join(trained_model_folder, f'fold_{f}', 'validation')
        if not isdir(expected_validation_folder):
            raise RuntimeError(f"fold {f} of model {trained_model_folder} is missing. Please train it!")
        predicted_files = subfiles(expected_validation_folder, suffix=dataset_json['file_ending'], join=False)
        for pf in predicted_files:
            if overwrite and isfile(join(merged_output_folder, pf)):
                raise RuntimeError(f'More than one of your folds has a prediction for case {pf}')
            if overwrite or not isfile(join(merged_output_folder, pf)):
                shutil.copy(join(expected_validation_folder, pf), join(merged_output_folder, pf))
                did_we_copy_something = True

    if did_we_copy_something or not isfile(join(merged_output_folder, 'summary.json')):
        label_manager = plans_manager.get_label_manager(dataset_json)
        compute_metrics_on_folder(join(nnUNet_raw, plans_manager.dataset_name, 'labelsTr'),
                                  merged_output_folder,
                                  join(merged_output_folder, 'summary.json'),
                                  rw,
                                  dataset_json['file_ending'],
                                  label_manager.foreground_regions if label_manager.has_regions else
                                  label_manager.foreground_labels,
                                  label_manager.ignore_label,
                                  num_processes)


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
    plans_manager = PlansManager(join(trained_model_folder, 'plans.json'))
    configuration_manager = plans_manager.get_configuration(configuration_name)
    if 'previous_stage' in plans_manager.available_configurations:
        prev_stage = configuration_manager.previous_stage_name
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
                            allowed_trained_models: Union[List[dict], Tuple[dict, ...]] = default_trained_models,
                            allow_ensembling: bool = True,
                            num_processes: int = default_num_processes,
                            overwrite: bool = True,
                            folds: Union[List[int], Tuple[int, ...]] = (0, 1, 2, 3, 4),
                            strict: bool = False):
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    all_results = {}

    allowed_trained_models = filter_available_models(deepcopy(allowed_trained_models), dataset_name_or_id)

    for m in allowed_trained_models:
        output_folder = get_output_folder(dataset_name_or_id, m['trainer'], m['plans'], m['configuration'], fold=None)
        if not isdir(output_folder) and strict:
            raise RuntimeError(f'{dataset_name}: The output folder of plans {m["plans"]} configuration '
                               f'{m["configuration"]} is missing. Please train the model (all requested folds!) first!')
        identifier = os.path.basename(output_folder)
        merged_output_folder = join(output_folder, f'crossval_results_folds_{folds_tuple_to_string(folds)}')
        accumulate_cv_results(output_folder, merged_output_folder, folds, num_processes, overwrite)
        all_results[identifier] = {
            'source': merged_output_folder,
            'result': load_summary_json(join(merged_output_folder, 'summary.json'))['foreground_mean']['Dice']
        }

    if allow_ensembling:
        for i in range(len(allowed_trained_models)):
            for j in range(i + 1, len(allowed_trained_models)):
                m1, m2 = allowed_trained_models[i], allowed_trained_models[j]

                output_folder_1 = get_output_folder(dataset_name_or_id, m1['trainer'], m1['plans'], m1['configuration'], fold=None)
                output_folder_2 = get_output_folder(dataset_name_or_id, m2['trainer'], m2['plans'], m2['configuration'], fold=None)
                identifier = get_ensemble_name(output_folder_1, output_folder_2, folds)

                output_folder_ensemble = join(nnUNet_results, dataset_name, 'ensembles', identifier)

                ensemble_crossvalidations([output_folder_1, output_folder_2], output_folder_ensemble, folds,
                                          num_processes, overwrite=overwrite)

                # evaluate ensembled predictions
                plans_manager = PlansManager(join(output_folder_1, 'plans.json'))
                dataset_json = load_json(join(output_folder_1, 'dataset.json'))
                label_manager = plans_manager.get_label_manager(dataset_json)
                rw = plans_manager.image_reader_writer_class()

                compute_metrics_on_folder(join(nnUNet_raw, dataset_name, 'labelsTr'),
                                          output_folder_ensemble,
                                          join(output_folder_ensemble, 'summary.json'),
                                          rw,
                                          dataset_json['file_ending'],
                                          label_manager.foreground_regions if label_manager.has_regions else
                                          label_manager.foreground_labels,
                                          label_manager.ignore_label,
                                          num_processes)
                all_results[identifier] = \
                    {
                    'source': output_folder_ensemble,
                    'result': load_summary_json(join(output_folder_ensemble, 'summary.json'))['foreground_mean']['Dice']
                    }

    # pick best and report inference command
    best_score = max([i['result'] for i in all_results.values()])
    best_keys = [k for k in all_results.keys() if all_results[k]['result'] == best_score]  # may never happen but theoretically
    # there can be a tie. Let's pick the first model in this case because it's going to be the simpler one (ensembles
    # come after single configs)
    best_key = best_keys[0]

    print('All results:')
    for k, v in all_results.items():
        print(f'{k}: {v}')
    print(f'\nBest: {best_key}: {all_results[best_key]["result"]}')

    print('You should now run nnUNetv2_determine_postprocessing:')
    print(f"nnUNetv2_determine_postprocessing -i {all_results[best_key]['source']} -ref {join(nnUNet_raw, dataset_name, 'labelsTr')} -np {num_processes}")

    # convert best key to inference command:
    print('\nAfter that you can use this for inference:')
    if best_key.startswith('ensemble___'):
        print('An ensemble won! What a surprise!')
        prefix, m1, m2, folds_string = best_key.split('___')
        tr1, pl1, c1 = convert_identifier_to_trainer_plans_config(m1)
        tr2, pl2, c2 = convert_identifier_to_trainer_plans_config(m2)
        print(generate_inference_command(dataset_name_or_id, c1, pl1, tr1, folds, save_npz=True, output_folder='OUTPUT_FOLDER_MODEL_1'))
        print(generate_inference_command(dataset_name_or_id, c2, pl2, tr2, folds, save_npz=True, output_folder='OUTPUT_FOLDER_MODEL_2'))
        print('Now run ensembling with:')
        print(f"nnUNetv2_ensemble -i OUTPUT_FOLDER_MODEL_1 OUTPUT_FOLDER_MODEL_2 -o OUTPUT_FOLDER_PP "
              f"-np {default_num_processes}")
    else:
        tr, pl, c = convert_identifier_to_trainer_plans_config(best_key)
        print(generate_inference_command(dataset_name_or_id, c, pl, tr, folds))

    print("Then apply the postprocessing with")
    print(f"nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP "
          f"-pp_pkl_file {join(all_results[best_key]['source'], 'postprocessing.pkl')} -np {num_processes}")


def dumb_trainer_config_plans_to_trained_models_dict(trainers: List[str], configs: List[str], plans: List[str]):
    ret = []
    for t in trainers:
        for c in configs:
            for p in plans:
                ret.append(
                    {'plans': p, 'configuration': c, 'trainer': t}
                )
    return tuple(ret)


def find_best_configuration_entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str, help='Dataset Name or id')
    parser.add_argument('-p', nargs='+', required=False, default=['nnUNetPlans'],
                        help='List of plan identifiers. Default: nnUNetPlans')
    parser.add_argument('-c', nargs='+', required=False, default=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'],
                        help="List of configurations. Default: ['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']")
    parser.add_argument('-tr', nargs='+', required=False, default=['nnUNetTrainer'],
                        help='List of trainers. Default: nnUNetTrainer')
    parser.add_argument('-np', required=False, default=default_num_processes, type=int,
                        help='Number of processes to use for ensembling, postprocessing etc')
    parser.add_argument('-f', nargs='+', type=int, default=(0, 1, 2, 3, 4),
                        help='Folds to use. Default: 0 1 2 3 4')
    parser.add_argument('--disable_ensembling', action='store_true', required=False,
                        help='Set this flag to disable ensembling')
    parser.add_argument('--no_overwrite', action='store_true',
                        help='If set we will not overwrite already ensembled files etc. May speed up concecutive '
                             'runs of this command (why would oyu want to do that?) at the risk of not updating '
                             'outdated results.')
    args = parser.parse_args()

    model_dict = dumb_trainer_config_plans_to_trained_models_dict(args.tr, args.c, args.p)
    dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)

    find_best_configuration(dataset_name, model_dict, allow_ensembling=not args.disable_ensembling,
                            num_processes=args.np, overwrite=not args.no_overwrite, folds=args.f,
                            strict=False)


def accumulate_crossval_results_entry_point():
    parser = argparse.ArgumentParser('Copies all predicted segmentations from the individual folds into one joint '
                                     'folder and evaluates them')
    parser.add_argument('dataset_name_or_id', type=str, help='Dataset Name or id')
    parser.add_argument('-c', type=str, required=True,
                        default='3d_fullres',
                        help="Configuration")
    parser.add_argument('-o', type=str, required=False, default=None,
                        help="Output folder. If not specified, the output folder will be located in the trained " \
                             "model directory (named crossval_results_folds_XXX).")
    parser.add_argument('-f', nargs='+', type=int, default=(0, 1, 2, 3, 4),
                        help='Folds to use. Default: 0 1 2 3 4')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plan identifier in which to search for the specified configuration. Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='Trainer class. Default: nnUNetTrainer')
    args = parser.parse_args()
    trained_model_folder = get_output_folder(args.dataset_name_or_id, args.tr, args.p, args.c)

    if args.o is None:
        merged_output_folder = join(trained_model_folder, f'crossval_results_folds_{folds_tuple_to_string(args.f)}')
    else:
        merged_output_folder = args.o

    accumulate_cv_results(trained_model_folder, merged_output_folder, args.f)


if __name__ == '__main__':
    find_best_configuration(4,
                            default_trained_models,
                            True,
                            8,
                            False,
                            (0, 1, 2, 3, 4))
