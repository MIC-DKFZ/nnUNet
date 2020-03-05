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

from batchgenerators.utilities.file_and_folder_operations import *
import os
from nnunet.evaluation.model_selection.summarize_results_in_one_json import summarize
from nnunet.paths import network_training_output_dir
import numpy as np


def list_to_string(l, delim=","):
    st = "%03.3f" % l[0]
    for i in l[1:]:
        st += delim + "%03.3f" % i
    return st


def write_plans_to_file(f, plans_file, stage=0, do_linebreak_at_end=True, override_name=None):
    a = load_pickle(plans_file)
    stages = list(a['plans_per_stage'].keys())
    stages.sort()
    patch_size_in_mm = [i * j for i, j in zip(a['plans_per_stage'][stages[stage]]['patch_size'],
                                              a['plans_per_stage'][stages[stage]]['current_spacing'])]
    median_patient_size_in_mm = [i * j for i, j in zip(a['plans_per_stage'][stages[stage]]['median_patient_size_in_voxels'],
                                              a['plans_per_stage'][stages[stage]]['current_spacing'])]
    if override_name is None:
        f.write(plans_file.split("/")[-2] + "__" + plans_file.split("/")[-1])
    else:
        f.write(override_name)
    f.write(";%d" % stage)
    f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['batch_size']))
    f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['num_pool_per_axis']))
    f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['patch_size']))
    f.write(";%s" % list_to_string(patch_size_in_mm))
    f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['median_patient_size_in_voxels']))
    f.write(";%s" % list_to_string(median_patient_size_in_mm))
    f.write(";%s" % list_to_string(a['plans_per_stage'][stages[stage]]['current_spacing']))
    f.write(";%s" % list_to_string(a['plans_per_stage'][stages[stage]]['original_spacing']))
    f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['pool_op_kernel_sizes']))
    f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['conv_kernel_sizes']))
    if do_linebreak_at_end:
        f.write("\n")


if __name__ == "__main__":
    summarize((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24, 27), output_dir=join(network_training_output_dir, "summary_fold0"), folds=(0,))
    base_dir = os.environ['RESULTS_FOLDER']
    nnunets = ['nnUNetV2', 'nnUNetV2_zspacing']
    task_ids = list(range(99))
    with open("summary.csv", 'w') as f:
        f.write("identifier;stage;batch_size;num_pool_per_axis;patch_size;patch_size(mm);median_patient_size_in_voxels;median_patient_size_in_mm;current_spacing;original_spacing;pool_op_kernel_sizes;conv_kernel_sizes;patient_dc;global_dc\n")
        for i in task_ids:
            for nnunet in nnunets:
                try:
                    summary_folder = join(base_dir, nnunet, "summary_fold0")
                    if isdir(summary_folder):
                        summary_files = subfiles(summary_folder, join=False, prefix="Task%03.0d_" % i, suffix=".json", sort=True)
                        for s in summary_files:
                            tmp = s.split("__")
                            trainer = tmp[2]

                            expected_output_folder = join(base_dir, nnunet, tmp[1], tmp[0], tmp[2].split(".")[0])
                            name = tmp[0] + "__" + nnunet + "__" + tmp[1] + "__" + tmp[2].split(".")[0]
                            global_dice_json = join(base_dir, nnunet, tmp[1], tmp[0], tmp[2].split(".")[0], "fold_0", "validation_tiledTrue_doMirror_True", "global_dice.json")

                            if not isdir(expected_output_folder) or len(tmp) > 3:
                                if len(tmp) == 2:
                                    continue
                                expected_output_folder = join(base_dir, nnunet, tmp[1], tmp[0], tmp[2] + "__" + tmp[3].split(".")[0])
                                name = tmp[0] + "__" + nnunet + "__" + tmp[1] + "__" + tmp[2] + "__" + tmp[3].split(".")[0]
                                global_dice_json = join(base_dir, nnunet, tmp[1], tmp[0], tmp[2] + "__" + tmp[3].split(".")[0], "fold_0", "validation_tiledTrue_doMirror_True", "global_dice.json")

                            assert isdir(expected_output_folder), "expected output dir not found"
                            plans_file = join(expected_output_folder, "plans.pkl")
                            assert isfile(plans_file)

                            plans = load_pickle(plans_file)
                            num_stages = len(plans['plans_per_stage'])
                            if num_stages > 1 and tmp[1] == "3d_fullres":
                                stage = 1
                            elif (num_stages == 1 and tmp[1] == "3d_fullres") or tmp[1] == "3d_lowres":
                                stage = 0
                            else:
                                print("skipping", s)
                                continue

                            g_dc = load_json(global_dice_json)
                            mn_glob_dc = np.mean(list(g_dc.values()))

                            write_plans_to_file(f, plans_file, stage, False, name)
                            # now read and add result to end of line
                            results = load_json(join(summary_folder, s))
                            mean_dc = results['results']['mean']['mean']['Dice']
                            f.write(";%03.3f" % mean_dc)
                            f.write(";%03.3f\n" % mn_glob_dc)
                            print(name, mean_dc)
                except Exception as e:
                    print(e)
