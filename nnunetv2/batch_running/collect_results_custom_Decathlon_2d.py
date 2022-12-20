from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.batch_running.collect_results_custom_Decathlon import collect_results, summarize
from nnunetv2.paths import nnUNet_results

if __name__ == '__main__':
    use_these_trainers = {
        'nnUNetTrainer': ('nnUNetPlans', ),
        'nnUNetTrainer_HRNet18': ('nnUNetPlans',),
        'nnUNetTrainer_HRNet32': ('nnUNetPlans',),
        'nnUNetTrainer_HRNet48': ('nnUNetPlans',),
        'nnUNetTrainerRMILoss': ('nnUNetPlans',),
        'anon_nnUNetTrainer1_dord0_sord0': ('nnUNetPlans',),
        'anon_nnUNetTrainer1_dord1_sord1': ('nnUNetPlans',),
    }
    all_results_file = join(nnUNet_results, 'hrnet_results.csv')
    datasets = [2, 3, 4, 17, 20, 24, 27, 38, 55, 64, 82]
    collect_results(use_these_trainers, datasets, all_results_file)

    folds = (0, )
    configs = ('2d', )
    output_file = join(nnUNet_results, 'hrnet_results_summary_fold0.csv')
    summarize(all_results_file, output_file, folds, configs, datasets, use_these_trainers)

