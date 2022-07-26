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
from typing import Union

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


def convert_id_to_dataset_name(dataset_id: int):
    startswith = "Dataset%03.0d" % dataset_id
    if nnUNet_preprocessed is not None and isdir(nnUNet_preprocessed):
        candidates_preprocessed = subdirs(nnUNet_preprocessed, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if nnUNet_raw is not None and isdir(nnUNet_raw):
        candidates_raw = subdirs(nnUNet_raw, prefix=startswith, join=False)
    else:
        candidates_raw = []

    candidates_trained_models = []
    if nnUNet_results is not None and isdir(nnUNet_results):
        candidates_trained_models += subdirs(nnUNet_results, prefix=startswith, join=False)

    all_candidates = candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one dataset name found for dataset id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (dataset_id, nnUNet_raw, nnUNet_preprocessed, nnUNet_results))
    if len(unique_candidates) == 0:
        raise RuntimeError("Could not find a dataset with the ID %d. Make sure the requested dataset ID exists and that "
                           "nnU-Net knows where raw and preprocessed data are located (see Documentation - "
                           "Installation). Here are your currently defined folders:\nnnUNet_preprocessed=%s\nRESULTS_"
                           "FOLDER=%s\nnnUNet_raw_data_base=%s\nIf something is not right, adapt your environemnt "
                           "variables." %
                           (dataset_id,
                            os.environ.get('nnUNet_preprocessed') if os.environ.get('nnUNet_preprocessed') is not None else 'None',
                            os.environ.get('RESULTS_FOLDER') if os.environ.get('RESULTS_FOLDER') is not None else 'None',
                            os.environ.get('nnUNet_raw') if os.environ.get('nnUNet_raw') is not None else 'None',
                            ))
    return unique_candidates[0]


def convert_dataset_name_to_id(dataset_name: str):
    assert dataset_name.startswith("Dataset")
    dataset_id = int(dataset_name[7:10])
    return dataset_id


def maybe_convert_to_dataset_name(dataset_name_or_id: Union[int, str]) -> str:
    if isinstance(dataset_name_or_id, str) and dataset_name_or_id.startswith("Dataset"):
        return dataset_name_or_id
    if isinstance(dataset_name_or_id, str):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id was a string and did not start with 'Dataset' so we tried to "
                             "convert it to a dataset ID (int). That failed, however. Please give an integer number "
                             "('1', '2', etc) or a correct tast name. Your input: %s" % dataset_name_or_id)
    return convert_id_to_dataset_name(dataset_name_or_id)