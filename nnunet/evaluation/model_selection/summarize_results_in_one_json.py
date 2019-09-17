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

from collections import OrderedDict
from nnunet.evaluation.add_mean_dice_to_json import foreground_mean
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import network_training_output_dir
import numpy as np


def summarize(tasks, models=('2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'),
              output_dir=join(network_training_output_dir, "summary_jsons"), folds=(0, 1, 2, 3, 4)):
    maybe_mkdir_p(output_dir)

    if len(tasks) == 1 and tasks[0] == "all":
        tasks = list(range(100))
    else:
        tasks = [int(i) for i in tasks]

    for model in models:
        for t in tasks:
            t = int(t)
            if not isdir(join(network_training_output_dir, model)):
                continue
            task_name = subfolders(join(network_training_output_dir, model), prefix="Task%02.0d" % t, join=False)
            if len(task_name) != 1:
                print("did not find unique output folder for network %s and task %s" % (model, t))
                continue
            task_name = task_name[0]
            out_dir_task = join(network_training_output_dir, model, task_name)

            model_trainers = subdirs(out_dir_task, join=False)
            for trainer in model_trainers:
                if trainer.startswith("fold"):
                    continue
                out_dir = join(out_dir_task, trainer)

                validation_folders = []
                for fld in folds:
                    d = join(out_dir, "fold%d"%fld)
                    if not isdir(d):
                        d = join(out_dir, "fold_%d"%fld)
                        if not isdir(d):
                            break
                    validation_folders += subfolders(d, prefix="validation", join=False)

                for v in validation_folders:
                    ok = True
                    metrics = OrderedDict()
                    for fld in folds:
                        d = join(out_dir, "fold%d"%fld)
                        if not isdir(d):
                            d = join(out_dir, "fold_%d"%fld)
                            if not isdir(d):
                                ok = False
                                break
                        validation_folder = join(d, v)

                        if not isfile(join(validation_folder, "summary.json")):
                            print("summary.json missing for net %s task %s fold %d" % (model, task_name, fld))
                            ok = False
                            break

                        metrics_tmp = load_json(join(validation_folder, "summary.json"))["results"]["mean"]
                        for l in metrics_tmp.keys():
                            if metrics.get(l) is None:
                                metrics[l] = OrderedDict()
                            for m in metrics_tmp[l].keys():
                                if metrics[l].get(m) is None:
                                    metrics[l][m] = []
                                metrics[l][m].append(metrics_tmp[l][m])
                    if ok:
                        for l in metrics.keys():
                            for m in metrics[l].keys():
                                assert len(metrics[l][m]) == len(folds)
                                metrics[l][m] = np.mean(metrics[l][m])
                        json_out = OrderedDict()
                        json_out["results"] = OrderedDict()
                        json_out["results"]["mean"] = metrics
                        json_out["task"] = task_name
                        json_out["description"] = model + " " + task_name + " all folds summary"
                        json_out["name"] = model + " " + task_name + " all folds summary"
                        json_out["experiment_name"] = model
                        save_json(json_out, join(out_dir, "summary_allFolds__%s.json" % v))
                        save_json(json_out, join(output_dir, "%s__%s__%s__%s.json" % (task_name, model, trainer, v)))
                        foreground_mean(join(out_dir, "summary_allFolds__%s.json" % v))
                        foreground_mean(join(output_dir, "%s__%s__%s__%s.json" % (task_name, model, trainer, v)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage="This is intended to identify the best model based on the five fold "
                                           "cross-validation. Running this script requires alle models to have been run "
                                           "already. This script will summarize the results of the five folds of all "
                                           "models in one json each for easy interpretability")
    parser.add_argument("-t", '--task_ids', nargs="+", required=True, help="task id. can be 'all'")
    parser.add_argument("-f", '--folds', nargs="+", required=False, type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("-m", '--models', nargs="+", required=False, default=['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'])

    args = parser.parse_args()
    tasks = args.task_ids
    models = args.models

    folds = args.folds
    summarize(tasks, models, folds=folds)

