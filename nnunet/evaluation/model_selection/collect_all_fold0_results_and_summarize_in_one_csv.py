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

from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, subfiles

from nnunet.evaluation.model_selection.summarize_results_in_one_json import summarize
from nnunet.paths import network_training_output_dir

if __name__ == "__main__":
    summary_output_folder = join(network_training_output_dir, "summary_jsons_fold0")
    maybe_mkdir_p(summary_output_folder)
    summarize(range(50), output_dir=summary_output_folder, folds=(0,))

    results_csv = join(network_training_output_dir, "summary_fold0.csv")

    summary_files = subfiles(summary_output_folder, suffix='.json', join=False)

    with open(results_csv, 'w') as f:
        for s in summary_files:
            if s.find("ensemble") == -1:
                task, network, trainer, plans, validation_folder = s.split("__")
            else:
                n1, n2 = s.split("--")
                n1 = n1[n1.find("ensemble_") + len("ensemble_") :]
                task = s.split("__")[0]
                network = "ensemble"
                trainer = n1
                plans = n2
                validation_folder = "none"
            validation_folder = validation_folder[:-len('.json')]
            results = load_json(join(summary_output_folder, s))['results']['mean']['mean']['Dice']
            f.write("%s,%s,%s,%s,%s,%02.4f\n" % (task,
                                            network, trainer, validation_folder, plans, results))

    summary_output_folder = join(network_training_output_dir, "summary_jsons")
    maybe_mkdir_p(summary_output_folder)
    summarize(['all'], output_dir=summary_output_folder)

    results_csv = join(network_training_output_dir, "summary_allFolds.csv")

    summary_files = subfiles(summary_output_folder, suffix='.json', join=False)

    with open(results_csv, 'w') as f:
        for s in summary_files:
            if s.find("ensemble") == -1:
                task, network, trainer, plans, validation_folder = s.split("__")
            else:
                n1, n2 = s.split("--")
                n1 = n1[n1.find("ensemble_") + len("ensemble_") :]
                task = s.split("__")[0]
                network = "ensemble"
                trainer = n1
                plans = n2
                validation_folder = "none"
            validation_folder = validation_folder[:-len('.json')]
            results = load_json(join(summary_output_folder, s))['results']['mean']['mean']['Dice']
            f.write("%s,%s,%s,%s,%s,%02.4f\n" % (task,
                                            network, trainer, validation_folder, plans, results))

