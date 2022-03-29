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


from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir, nnUNet_cropped_data, network_training_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
import numpy as np


def convert_id_to_task_name(task_id: int):
    startswith = "Task%03.0d" % task_id
    if preprocessing_output_dir is not None:
        candidates_preprocessed = subdirs(preprocessing_output_dir, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if nnUNet_raw_data is not None:
        candidates_raw = subdirs(nnUNet_raw_data, prefix=startswith, join=False)
    else:
        candidates_raw = []

    if nnUNet_cropped_data is not None:
        candidates_cropped = subdirs(nnUNet_cropped_data, prefix=startswith, join=False)
    else:
        candidates_cropped = []

    candidates_trained_models = []
    if network_training_output_dir is not None:
        for m in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']:
            if isdir(join(network_training_output_dir, m)):
                candidates_trained_models += subdirs(join(network_training_output_dir, m), prefix=startswith, join=False)

    all_candidates = candidates_cropped + candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one task name found for task id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (task_id, nnUNet_raw_data, preprocessing_output_dir,
                                                               nnUNet_cropped_data))
    if len(unique_candidates) == 0:
        raise RuntimeError("Could not find a task with the ID %d. Make sure the requested task ID exists and that "
                           "nnU-Net knows where raw and preprocessed data are located (see Documentation - "
                           "Installation). Here are your currently defined folders:\nnnUNet_preprocessed=%s\nRESULTS_"
                           "FOLDER=%s\nnnUNet_raw_data_base=%s\nIf something is not right, adapt your environemnt "
                           "variables." %
                           (task_id,
                            os.environ.get('nnUNet_preprocessed') if os.environ.get('nnUNet_preprocessed') is not None else 'None',
                            os.environ.get('RESULTS_FOLDER') if os.environ.get('RESULTS_FOLDER') is not None else 'None',
                            os.environ.get('nnUNet_raw_data_base') if os.environ.get('nnUNet_raw_data_base') is not None else 'None',
                            ))
    return unique_candidates[0]


def convert_task_name_to_id(task_name: str):
    assert task_name.startswith("Task")
    task_id = int(task_name[4:7])
    return task_id
