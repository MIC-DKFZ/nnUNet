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
import shutil
from itertools import combinations
import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
from nnunet.evaluation.add_mean_dice_to_json import foreground_mean
from nnunet.evaluation.evaluator import evaluate_folder
from nnunet.evaluation.model_selection.ensemble import ensemble
from nnunet.paths import network_training_output_dir
import numpy as np
from subprocess import call
from nnunet.postprocessing.consolidate_postprocessing import consolidate_folds, collect_cv_niftis
from nnunet.utilities.folder_names import get_output_folder_name
from nnunet.paths import default_cascade_trainer, default_trainer, default_plans_identifier


def find_task_name(folder, task_id):
    candidates = subdirs(folder, prefix="Task%03.0d_" % task_id, join=False)
    assert len(candidates) > 0, "no candidate for Task id %d found in folder %s" % (task_id, folder)
    assert len(candidates) == 1, "more than one candidate for Task id %d found in folder %s" % (task_id, folder)
    return candidates[0]


def get_mean_foreground_dice(json_file):
    results = load_json(json_file)
    return get_foreground_mean(results)


def get_foreground_mean(results):
    results_mean = results['results']['mean']
    dice_scores = [results_mean[i]['Dice'] for i in results_mean.keys() if i != "0" and i != 'mean']
    return np.mean(dice_scores)


def main():
    import argparse
    parser = argparse.ArgumentParser(usage="This is intended to identify the best model based on the five fold "
                                           "cross-validation. Running this script requires all models to have been run "
                                           "already. This script will summarize the results of the five folds of all "
                                           "models in one json each for easy interpretability")

    parser.add_argument("-m", '--models', nargs="+", required=False, default=['2d', '3d_lowres', '3d_fullres',
                                                                              '3d_cascade_fullres'])
    parser.add_argument("-t", '--task_ids', nargs="+", required=True)

    parser.add_argument("-tr", type=str, required=False, default=default_trainer,
                        help="nnUNetTrainer class. Default: %s" % default_trainer)
    parser.add_argument("-ctr", type=str, required=False, default=default_cascade_trainer,
                        help="nnUNetTrainer class for cascade model. Default: %s" % default_cascade_trainer)
    parser.add_argument("-pl", type=str, required=False, default=default_plans_identifier,
                        help="plans name, Default: %s" % default_plans_identifier)
    parser.add_argument('-f', '--folds', nargs='+', default=(0, 1, 2, 3, 4), help="Use this if you have non-standard "
                                                                                  "folds. Experienced users only.")
    parser.add_argument('--disable_ensembling', required=False, default=False, action='store_true',
                        help='Set this flag to disable the use of ensembling. This will find the best single '
                             'configuration for each task.')
    parser.add_argument("--disable_postprocessing", required=False, default=False, action="store_true",
                        help="Set this flag if you want to disable the use of postprocessing")

    args = parser.parse_args()
    tasks = [int(i) for i in args.task_ids]

    models = args.models
    tr = args.tr
    trc = args.ctr
    pl = args.pl
    disable_ensembling = args.disable_ensembling
    disable_postprocessing = args.disable_postprocessing
    folds = tuple(int(i) for i in args.folds)

    validation_folder = "validation_raw"

    # this script now acts independently from the summary jsons. That was unnecessary
    id_task_mapping = {}

    for t in tasks:
        # first collect pure model performance
        results = {}
        all_results = {}
        valid_models = []
        for m in models:
            if m == "3d_cascade_fullres":
                trainer = trc
            else:
                trainer = tr

            if t not in id_task_mapping.keys():
                task_name = find_task_name(get_output_folder_name(m), t)
                id_task_mapping[t] = task_name

            output_folder = get_output_folder_name(m, id_task_mapping[t], trainer, pl)
            if not isdir(output_folder):
                raise RuntimeError("Output folder for model %s is missing, expected: %s" % (m, output_folder))

            if disable_postprocessing:
                # we need to collect the predicted niftis from the 5-fold cv and evaluate them against the ground truth
                cv_niftis_folder = join(output_folder, 'cv_niftis_raw')

                if not isfile(join(cv_niftis_folder, 'summary.json')):
                    print(t, m, ': collecting niftis from 5-fold cv')
                    if isdir(cv_niftis_folder):
                        shutil.rmtree(cv_niftis_folder)

                    collect_cv_niftis(output_folder, cv_niftis_folder, validation_folder, folds)

                    niftis_gt = subfiles(join(output_folder, "gt_niftis"), suffix='.nii.gz', join=False)
                    niftis_cv = subfiles(cv_niftis_folder, suffix='.nii.gz', join=False)
                    if not all([i in niftis_gt for i in niftis_cv]):
                        raise AssertionError("It does not seem like you trained all the folds! Train " \
                                             "all folds first! There are %d gt niftis in %s but only " \
                                             "%d predicted niftis in %s" % (len(niftis_gt), niftis_gt,
                                                                            len(niftis_cv), niftis_cv))

                    # load a summary file so that we can know what class labels to expect
                    summary_fold0 = load_json(join(output_folder, "fold_%d" % folds[0], validation_folder,
                                                   "summary.json"))['results']['mean']
                    # read classes from summary.json
                    classes = tuple((int(i) for i in summary_fold0.keys()))

                    # evaluate the cv niftis
                    print(t, m, ': evaluating 5-fold cv results')
                    evaluate_folder(join(output_folder, "gt_niftis"), cv_niftis_folder, classes)

            else:
                postprocessing_json = join(output_folder, "postprocessing.json")
                cv_niftis_folder = join(output_folder, "cv_niftis_raw")

                # we need cv_niftis_postprocessed to know the single model performance. And we need the
                # postprocessing_json. If either of those is missing, rerun consolidate_folds
                if not isfile(postprocessing_json) or not isdir(cv_niftis_folder):
                    print("running missing postprocessing for %s and model %s" % (id_task_mapping[t], m))
                    consolidate_folds(output_folder, folds=folds)

                assert isfile(postprocessing_json), "Postprocessing json missing, expected: %s" % postprocessing_json
                assert isdir(cv_niftis_folder), "Folder with niftis from CV missing, expected: %s" % cv_niftis_folder

            # obtain mean foreground dice
            summary_file = join(cv_niftis_folder, "summary.json")
            results[m] = get_mean_foreground_dice(summary_file)
            foreground_mean(summary_file)
            all_results[m] = load_json(summary_file)['results']['mean']
            valid_models.append(m)

        if not disable_ensembling:
            # now run ensembling and add ensembling to results
            print("\nI will now ensemble combinations of the following models:\n", valid_models)
            if len(valid_models) > 1:
                for m1, m2 in combinations(valid_models, 2):

                    trainer_m1 = trc if m1 == "3d_cascade_fullres" else tr
                    trainer_m2 = trc if m2 == "3d_cascade_fullres" else tr

                    ensemble_name = "ensemble_" + m1 + "__" + trainer_m1 + "__" + pl + "--" + m2 + "__" + trainer_m2 + "__" + pl
                    output_folder_base = join(network_training_output_dir, "ensembles", id_task_mapping[t], ensemble_name)
                    maybe_mkdir_p(output_folder_base)

                    network1_folder = get_output_folder_name(m1, id_task_mapping[t], trainer_m1, pl)
                    network2_folder = get_output_folder_name(m2, id_task_mapping[t], trainer_m2, pl)

                    print("ensembling", network1_folder, network2_folder)
                    ensemble(network1_folder, network2_folder, output_folder_base, id_task_mapping[t], validation_folder, folds, allow_ensembling=not disable_postprocessing)
                    # ensembling will automatically do postprocessingget_foreground_mean

                    # now get result of ensemble
                    results[ensemble_name] = get_mean_foreground_dice(join(output_folder_base, "ensembled_raw", "summary.json"))
                    summary_file = join(output_folder_base, "ensembled_raw", "summary.json")
                    foreground_mean(summary_file)
                    all_results[ensemble_name] = load_json(summary_file)['results']['mean']

        # now print all mean foreground dice and highlight the best
        foreground_dices = list(results.values())
        best = np.max(foreground_dices)
        for k, v in results.items():
            print(k, v)

        predict_str = ""
        best_model = None
        for k, v in results.items():
            if v == best:
                print("%s submit model %s" % (id_task_mapping[t], k), v)
                best_model = k
                print("\nHere is how you should predict test cases. Run in sequential order and replace all input and output folder names with your personalized ones\n")
                if k.startswith("ensemble"):
                    tmp = k[len("ensemble_"):]
                    model1, model2 = tmp.split("--")
                    m1, t1, pl1 = model1.split("__")
                    m2, t2, pl2 = model2.split("__")
                    predict_str += "nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr " + tr + " -ctr " + trc + " -m " + m1 + " -p " + pl + " -t " + \
                                   id_task_mapping[t] + "\n"
                    predict_str += "nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL2 -tr " + tr + " -ctr " + trc + " -m " + m2 + " -p " + pl + " -t " + \
                                   id_task_mapping[t] + "\n"

                    if not disable_postprocessing:
                        predict_str += "nnUNet_ensemble -f OUTPUT_FOLDER_MODEL1 OUTPUT_FOLDER_MODEL2 -o OUTPUT_FOLDER -pp " + join(network_training_output_dir, "ensembles", id_task_mapping[t], k, "postprocessing.json") + "\n"
                    else:
                        predict_str += "nnUNet_ensemble -f OUTPUT_FOLDER_MODEL1 OUTPUT_FOLDER_MODEL2 -o OUTPUT_FOLDER\n"
                else:
                    predict_str += "nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr " + tr + " -ctr " + trc + " -m " + k + " -p " + pl + " -t " + \
                                   id_task_mapping[t] + "\n"
                print(predict_str)

        summary_folder = join(network_training_output_dir, "ensembles", id_task_mapping[t])
        maybe_mkdir_p(summary_folder)
        with open(join(summary_folder, "prediction_commands.txt"), 'w') as f:
            f.write(predict_str)

        num_classes = len([i for i in all_results[best_model].keys() if i != 'mean' and i != '0'])
        with open(join(summary_folder, "summary.csv"), 'w') as f:
            f.write("model")
            for c in range(1, num_classes + 1):
                f.write(",class%d" % c)
            f.write(",average")
            f.write("\n")
            for m in all_results.keys():
                f.write(m)
                for c in range(1, num_classes + 1):
                    f.write(",%01.4f" % all_results[m][str(c)]["Dice"])
                f.write(",%01.4f" % all_results[m]['mean']["Dice"])
                f.write("\n")


if __name__ == "__main__":
    main()
