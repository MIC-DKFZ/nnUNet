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
from nnunet.paths import network_training_output_dir


def get_output_folder_name(model: str, task: str = None, trainer: str = None, plans: str = None, fold: int = None,
                           overwrite_training_output_dir: str = None):
    """
    Retrieves the correct output directory for the nnU-Net model described by the input parameters

    :param model:
    :param task:
    :param trainer:
    :param plans:
    :param fold:
    :param overwrite_training_output_dir:
    :return:
    """
    assert model in ["2d", "3d_cascade_fullres", '3d_fullres', '3d_lowres']

    if overwrite_training_output_dir is not None:
        tr_dir = overwrite_training_output_dir
    else:
        tr_dir = network_training_output_dir

    current = join(tr_dir, model)
    if task is not None:
        current = join(current, task)
        if trainer is not None and plans is not None:
            current = join(current, trainer + "__" + plans)
            if fold is not None:
                current = join(current, "fold_%d" % fold)
    return current
