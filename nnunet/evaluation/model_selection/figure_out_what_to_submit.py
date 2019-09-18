#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import network_training_output_dir
import numpy as np
from nnunet.evaluation.add_mean_dice_to_json import foreground_mean
from subprocess import call
import SimpleITK as sitk
from nnunet.run.default_configuration import get_output_folder


def copy_nifti_and_convert_to_uint8(args):
    source_file, target_file = args
    i = sitk.ReadImage(source_file)
    j = sitk.GetImageFromArray(sitk.GetArrayFromImage(i).astype(np.uint8))
    j.CopyInformation(i)
    sitk.WriteImage(j, target_file)


if __name__ == "__main__":
    # This script was hacked together at some point and is ugly af. TODO this needs to be redone properly
    import argparse
    parser = argparse.ArgumentParser(usage="This is intended to identify the best model based on the five fold "
                                           "cross-validation. Running this script requires alle models to have been run "
                                           "already. This script will summarize the results of the five folds of all "
                                           "models in one json each for easy interpretability")
    parser.add_argument("-m", '--models', nargs="+", required=False, default=['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'])
    parser.add_argument("-t", '--task_ids', nargs="+", required=False, default=list(range(100)))
    parser.add_argument("-v", '--validation_folder', required=False, default="validation_raw")

    args = parser.parse_args()
    tasks = [int(i) for i in args.task_ids]

    models = args.models
    validation_folder = args.validation_folder

    out_dir_all_json = join(network_training_output_dir, "summary_jsons")

    json_files = [i for i in subfiles(out_dir_all_json, suffix=".json", join=True) if i.find("ensemble") == -1]

    # do mean over foreground
    for j in json_files:
        foreground_mean(j)

    # for each task, run ensembling using all combinations of two models
    for t in tasks:
        json_files_task = [i for i in subfiles(out_dir_all_json, prefix="Task%02.0d_" % t) if i.find("ensemble") == -1 and i.endswith(validation_folder + ".json")]
        if len(json_files_task) > 0:
            task_name = json_files_task[0].split("/")[-1].split("__")[0]
            print(task_name)

            for i in range(len(json_files_task) - 1):
                for j in range(i+1, len(json_files_task)):
                    # networks are stored as
                    # task__configuration__trainer__plans
                    network1 = json_files_task[i].split("/")[-1].split("__")
                    network1[-1] = network1[-1].split(".")[0]
                    task, configuration, trainer, plans_identifier, _ = network1
                    network1_folder = get_output_folder(configuration, task, trainer, plans_identifier)
                    name1 = configuration + "__" + trainer + "__" + plans_identifier

                    network2 = json_files_task[j].split("/")[-1].split("__")
                    network2[-1] = network2[-1].split(".")[0]
                    task, configuration, trainer, plans_identifier, _ = network2
                    network2_folder = get_output_folder(configuration, task, trainer, plans_identifier)
                    name2 = configuration + "__" + trainer + "__" + plans_identifier

                    if np.argsort((name1, name2))[0] == 1:
                        name1, name2 = name2, name1
                        network1_folder, network2_folder = network2_folder, network1_folder

                    output_folder = join(network_training_output_dir, "ensembles", task_name, "ensemble_" + name1 + "--" + name2)
                    # now ensemble
                    print(network1_folder, network2_folder)
                    p = call(["python", join(nnunet.__path__[0], "evaluation/model_selection/ensemble.py"), network1_folder, network2_folder, output_folder, task_name, validation_folder])

    # now rerun adding the mean foreground dice
    json_files = subfiles(out_dir_all_json, suffix=".json", join=True)

    # do mean over foreground
    for j in json_files:
        foreground_mean(j)

    # now load all json for each task and find best
    with open(join(network_training_output_dir, "use_this_for_test.csv"), 'w') as f:
        for t in tasks:
            t = int(t)
            json_files_task = subfiles(out_dir_all_json, prefix="Task%02.0d_" % t)
            if len(json_files_task) > 0:
                task_name = json_files_task[0].split("/")[-1].split("__")[0]
                print(task_name)
                mean_dice = []
                for j in json_files_task:
                    js = load_json(j)
                    mean_dice.append(js['results']['mean']['mean']['Dice'])
                best = np.argsort(mean_dice)[::-1][0]
                j = json_files_task[best].split("/")[-1]

                print("%s: submit model %s" % (task_name, j))
                f.write("%s,%s\n" % (task_name, j))
