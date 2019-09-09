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

import json
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles
from collections import OrderedDict


def foreground_mean(filename):
    with open(filename, 'r') as f:
        res = json.load(f)
    class_ids = np.array([int(i) for i in res['results']['mean'].keys() if (i != 'mean')])
    class_ids = class_ids[class_ids != 0]
    class_ids = class_ids[class_ids != -1]
    class_ids = class_ids[class_ids != 99]

    tmp = res['results']['mean'].get('99')
    if tmp is not None:
        _ = res['results']['mean'].pop('99')

    metrics = res['results']['mean']['1'].keys()
    res['results']['mean']["mean"] = OrderedDict()
    for m in metrics:
        foreground_values = [res['results']['mean'][str(i)][m] for i in class_ids]
        res['results']['mean']["mean"][m] = np.nanmean(foreground_values)
    with open(filename, 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)


def run_in_folder(folder):
    json_files = subfiles(folder, True, None, ".json", True)
    json_files = [i for i in json_files if not i.split("/")[-1].startswith(".") and not i.endswith("_globalMean.json")] # stupid mac
    for j in json_files:
        foreground_mean(j)


if __name__ == "__main__":
    folder = "/media/fabian/Results/nnUNetOutput_final/summary_jsons"
    run_in_folder(folder)
