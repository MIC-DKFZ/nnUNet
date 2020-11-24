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

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import preprocessing_output_dir


# This file is intended to double check nnUNets design choices. It is intended to be used for developent purposes only
def summarize_plans(file):
    plans = load_pickle(file)
    print("num_classes: ", plans['num_classes'])
    print("modalities: ", plans['modalities'])
    print("use_mask_for_norm", plans['use_mask_for_norm'])
    print("keep_only_largest_region", plans['keep_only_largest_region'])
    print("min_region_size_per_class", plans['min_region_size_per_class'])
    print("min_size_per_class", plans['min_size_per_class'])
    print("normalization_schemes", plans['normalization_schemes'])
    print("stages...\n")

    for i in range(len(plans['plans_per_stage'])):
        print("stage: ", i)
        print(plans['plans_per_stage'][i])
        print("")


def write_plans_to_file(f, plans_file):
    print(plans_file)
    a = load_pickle(plans_file)
    stages = list(a['plans_per_stage'].keys())
    stages.sort()
    for stage in stages:
        patch_size_in_mm = [i * j for i, j in zip(a['plans_per_stage'][stages[stage]]['patch_size'],
                                                  a['plans_per_stage'][stages[stage]]['current_spacing'])]
        median_patient_size_in_mm = [i * j for i, j in zip(a['plans_per_stage'][stages[stage]]['median_patient_size_in_voxels'],
                                                  a['plans_per_stage'][stages[stage]]['current_spacing'])]
        f.write(plans_file.split("/")[-2])
        f.write(";%s" % plans_file.split("/")[-1])
        f.write(";%d" % stage)
        f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['batch_size']))
        f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['num_pool_per_axis']))
        f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['patch_size']))
        f.write(";%s" % str([str("%03.2f" % i) for i in patch_size_in_mm]))
        f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['median_patient_size_in_voxels']))
        f.write(";%s" % str([str("%03.2f" % i) for i in median_patient_size_in_mm]))
        f.write(";%s" % str([str("%03.2f" % i) for i in a['plans_per_stage'][stages[stage]]['current_spacing']]))
        f.write(";%s" % str([str("%03.2f" % i) for i in a['plans_per_stage'][stages[stage]]['original_spacing']]))
        f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['pool_op_kernel_sizes']))
        f.write(";%s" % str(a['plans_per_stage'][stages[stage]]['conv_kernel_sizes']))
        f.write(";%s" % str(a['data_identifier']))
        f.write("\n")


if __name__ == "__main__":
    base_dir = './'#preprocessing_output_dir''
    task_dirs = [i for i in subdirs(base_dir, join=False, prefix="Task") if i.find("BrainTumor") == -1 and i.find("MSSeg") == -1]
    print("found %d tasks" % len(task_dirs))

    with open("2019_02_06_plans_summary.csv", 'w') as f:
        f.write("task;plans_file;stage;batch_size;num_pool_per_axis;patch_size;patch_size(mm);median_patient_size_in_voxels;median_patient_size_in_mm;current_spacing;original_spacing;pool_op_kernel_sizes;conv_kernel_sizes\n")
        for t in task_dirs:
            print(t)
            tmp = join(base_dir, t)
            plans_files = [i for i in subfiles(tmp, suffix=".pkl", join=False) if i.find("_plans_") != -1 and i.find("Dgx2") == -1]
            for p in plans_files:
                write_plans_to_file(f, join(tmp, p))
            f.write("\n")


