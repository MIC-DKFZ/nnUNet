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

import json
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles
import os
from collections import OrderedDict

folder = "/home/fabian/drives/E132-Projekte/Projects/2018_MedicalDecathlon/Leaderboard"
task_descriptors = ['2D final 2',
                    '2D final, less pool, dc and topK, fold0',
                    '2D final pseudo3d 7, fold0',
                    '2D final, less pool, dc and ce, fold0',
                    '3D stage0 final 2, fold0',
                    '3D fullres final 2, fold0']
task_ids_with_no_stage0 = ["Task001_BrainTumour", "Task004_Hippocampus", "Task005_Prostate"]

mean_scores = OrderedDict()
for t in task_descriptors:
    mean_scores[t] = OrderedDict()

json_files = subfiles(folder, True, None, ".json", True)
# json_files = [i for i in json_files if not i.split("/")[-1].startswith(".")]  # stupid mac
json_files = [i for i in json_files if not os.path.basename(i).startswith(".")]  # stupid mac
for j in json_files:
    with open(j, 'r') as f:
        res = json.load(f)
    task = res['task']
    if task != "Task999_ALL":
        name = res['name']
        if name in task_descriptors:
            if task not in list(mean_scores[name].keys()):
                mean_scores[name][task] = res['results']['mean']['mean']
            else:
                raise RuntimeError("duplicate task %s for description %s" % (task, name))

for t in task_ids_with_no_stage0:
    mean_scores["3D stage0 final 2, fold0"][t] = mean_scores["3D fullres final 2, fold0"][t]

a = set()
for i in mean_scores.keys():
    a = a.union(list(mean_scores[i].keys()))

for i in mean_scores.keys():
    try:
        for t in list(a):
            assert t in mean_scores[i].keys(), "did not find task %s for experiment %s" % (t, i)
        new_res = OrderedDict()
        new_res['name'] = i
        new_res['author'] = "Fabian"
        new_res['task'] = "Task999_ALL"
        new_res['results'] = OrderedDict()
        new_res['results']['mean'] = OrderedDict()
        new_res['results']['mean']['mean'] = OrderedDict()
        tasks = list(mean_scores[i].keys())
        metrics = mean_scores[i][tasks[0]].keys()
        for m in metrics:
            foreground_values = [mean_scores[i][n][m] for n in tasks]
            new_res['results']['mean']["mean"][m] = np.nanmean(foreground_values)
        output_fname = i.replace(" ", "_") + "_globalMean.json"
        with open(os.path.join(folder, output_fname), 'w') as f:
            json.dump(new_res, f)
    except AssertionError:
        print("could not process experiment %s" % i)
        print("did not find task %s for experiment %s" % (t, i))

