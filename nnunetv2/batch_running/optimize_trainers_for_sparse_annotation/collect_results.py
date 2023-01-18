from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.batch_running.collect_results_custom_Decathlon import summarize, collect_results

from nnunetv2.paths import nnUNet_results

if __name__ == '__main__':
    use_these_trainers = {
        'nnUNetTrainer': ('nnUNetPlans', ),
        'nnUNetTrainer_betterIgnoreSampling': ('nnUNetPlans', ),
        'nnUNetTrainer_DASegOrd0': ('nnUNetPlans', ),
        'nnUNetTrainer_ignoreLabel_fixCEAggr': ('nnUNetPlans', ),
        'nnUNetTrainer_ignoreLabel_lossBalancing': ('nnUNetPlans',),
    }
    configs = (
        "3d_lowres_sparse_patches_and_slices_10_2",
        "3d_lowres_sparse_randOrthSlices10",
        "3d_lowres_sparse_pixelwise10",
        "3d_lowres_sparse_patches_10_2"
    )

    all_results_file = join(nnUNet_results, 'sparse_annotations_rawResults.csv')
    datasets = [216, ]
    collect_results(use_these_trainers, datasets, all_results_file, configurations=configs)

    folds = (0, 1, 2, )

    output_file = join(nnUNet_results, 'sparse_annotations_summary.csv')
    summarize(all_results_file, output_file, folds, configs, datasets, use_these_trainers)

