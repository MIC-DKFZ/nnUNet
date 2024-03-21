from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.batch_running.collect_results_custom_Decathlon import collect_results, summarize
import nnunetv2.paths as paths

if __name__ == '__main__':
    use_these_trainers = {
        'nnUNetTrainer': ('nnUNetPlans', ),
    }
    all_results_file = join(paths.nnUNet_results, 'hrnet_results.csv')
    datasets = [2, 3, 4, 17, 20, 24, 27, 38, 55, 64, 82, 217]
    collect_results(use_these_trainers, datasets, all_results_file)

    folds = (0, )
    configs = ('2d', )
    output_file = join(paths.nnUNet_results, 'hrnet_results_summary_fold0.csv')
    summarize(all_results_file, output_file, folds, configs, datasets, use_these_trainers)

