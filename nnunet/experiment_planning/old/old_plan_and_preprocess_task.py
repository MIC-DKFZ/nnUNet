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

from nnunet.experiment_planning.utils import split_4d, crop, analyze_dataset, plan_and_preprocess
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, help="task name. There must be a matching folder in "
                                                       "raw_dataset_dir", required=True)
    parser.add_argument('-pl', '--processes_lowres', type=int, default=8, help='number of processes used for '
                                                                               'preprocessing 3d_lowres data, image '
                                                                               'splitting and image cropping '
                                                                               'Default: 8. The distinction between '
                                                                               'processes_lowres and processes_fullres '
                                                                               'is necessary because preprocessing '
                                                                               'at full resolution needs a lot of '
                                                                               'RAM', required=False)
    parser.add_argument('-pf', '--processes_fullres', type=int, default=8, help='number of processes used for '
                                                                                'preprocessing 2d and 3d_fullres '
                                                                                'data. Default: 3', required=False)
    parser.add_argument('-o', '--override', type=int, default=0, help="set this to 1 if you want to override "
                                                                      "cropped data and intensityproperties. Default: 0",
                        required=False)
    parser.add_argument('-s', '--use_splitted', type=int, default=1, help='1 = use splitted data if already present ('
                                                                          'skip split_4d). 0 = do splitting again. '
                                                                          'It is save to set this to 1 at all times '
                                                                          'unless the dataset was updated in the '
                                                                          'meantime. Default: 1', required=False)
    parser.add_argument('-no_preprocessing', type=int, default=0, help='debug only. If set to 1 this will run only'
                                                                       'experiment planning and not run the '
                                                                       'preprocessing')

    args = parser.parse_args()
    task = args.task
    processes_lowres = args.processes_lowres
    processes_fullres = args.processes_fullres
    override = args.override
    use_splitted = args.use_splitted
    no_preprocessing = args.no_preprocessing

    if override == 0:
        override = False
    elif override == 1:
        override = True
    else:
        raise ValueError("only 0 or 1 allowed for override")

    if no_preprocessing == 0:
        no_preprocessing = False
    elif no_preprocessing == 1:
        no_preprocessing = True
    else:
        raise ValueError("only 0 or 1 allowed for override")

    if use_splitted == 0:
        use_splitted = False
    elif use_splitted == 1:
        use_splitted = True
    else:
        raise ValueError("only 0 or 1 allowed for use_splitted")

    if task == "all":
        all_tasks = subdirs(nnUNet_raw_data, prefix="Task", join=False)
        for t in all_tasks:
            crop(t, override=override, num_threads=processes_lowres)
            analyze_dataset(t, override=override, collect_intensityproperties=True, num_processes=processes_lowres)
            plan_and_preprocess(t, processes_lowres, processes_fullres, no_preprocessing)
    else:
        if not use_splitted or not isdir(join(nnUNet_raw_data, task)):
            print("splitting task ", task)
            split_4d(task)

        crop(task, override=override, num_threads=processes_lowres)
        analyze_dataset(task, override, collect_intensityproperties=True, num_processes=processes_lowres)
        plan_and_preprocess(task, processes_lowres, processes_fullres, no_preprocessing)
