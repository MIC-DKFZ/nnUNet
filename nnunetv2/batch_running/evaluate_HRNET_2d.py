from typing import Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.batch_running.collect_results_custom_Decathlon import collect_results, summarize
from nnunetv2.evaluation.evaluate_predictions import load_summary_json
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name, convert_dataset_name_to_id
from nnunetv2.utilities.file_path_utilities import get_output_folder


if __name__ == '__main__':
    use_these_trainers = {
        'nnUNetTrainer': ('nnUNetPlans', ),
        'nnUNetTrainer_HRNet18': ('nnUNetPlans',),
        'nnUNetTrainer_HRNet32': ('nnUNetPlans',),
        'nnUNetTrainer_HRNet48': ('nnUNetPlans',),
    }
    all_results_file = join(nnUNet_results, 'hrnet_results.csv')
    datasets = [2, 3, 4, 17, 20, 24, 27, 38, 55, 64, 82]
    collect_results(use_these_trainers, datasets, all_results_file)

    folds = (0, )
    configs = ('2d', )
    output_file = join(nnUNet_results, 'hrnet_results_summary_fold0.csv')
    summarize(all_results_file, output_file, folds, configs, datasets, use_these_trainers)

