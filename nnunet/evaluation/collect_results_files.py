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

import os
import shutil
from batchgenerators.utilities.file_and_folder_operations import subdirs, subfiles


def crawl_and_copy(current_folder, out_folder, prefix="fabian_", suffix="ummary.json"):
    """
    This script will run recursively through all subfolders of current_folder and copy all files that end with
    suffix with some automatically generated prefix into out_folder
    :param current_folder:
    :param out_folder:
    :param prefix:
    :return:
    """
    s = subdirs(current_folder, join=False)
    f = subfiles(current_folder, join=False)
    f = [i for i in f if i.endswith(suffix)]
    if current_folder.find("fold0") != -1:
        for fl in f:
            shutil.copy(os.path.join(current_folder, fl), os.path.join(out_folder, prefix+fl))
    for su in s:
        if prefix == "":
            add = su
        else:
            add = "__" + su
        crawl_and_copy(os.path.join(current_folder, su), out_folder, prefix=prefix+add)


if __name__ == "__main__":
    from nnunet.paths import network_training_output_dir
    output_folder = "/home/fabian/PhD/results/nnUNetV2/leaderboard"
    crawl_and_copy(network_training_output_dir, output_folder)
    from nnunet.evaluation.add_mean_dice_to_json import run_in_folder
    run_in_folder(output_folder)
