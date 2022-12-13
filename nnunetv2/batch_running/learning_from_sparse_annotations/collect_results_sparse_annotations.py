import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.batch_running.collect_results_custom_Decathlon import collect_results
from nnunetv2.paths import nnUNet_results

if __name__ == '__main__':
    use_these_trainers = {
        'nnUNetTrainer': ('nnUNetPlans',),
        'nnUNetTrainerDA5': ('nnUNetPlans',),
        'nnUNetTrainerDA5ord0': ('nnUNetPlans',),
        'nnUNetTrainerDA5_betterIgnoreSampling': ('nnUNetPlans',),
        'nnUNetTrainer_betterIgnoreSampling': ('nnUNetPlans',),
    }

    configurations_all = {
        216: (
            "3d_lowres",
            "3d_lowres_sparse_slicewise_10",
            "3d_lowres_sparse_slicewise_30",
            "3d_lowres_sparse_randblobs",
            "3d_lowres_sparse_pixelwise",
            '3d_lowres_sparse_hybridsparsepatchesslices',
            '3d_lowres_sparse_sparsepatches',
            '3d_lowres_sparse_sliceOSfg10',
            '3d_lowres_sparse_slicewiserand10',
            '3d_lowres_sparse_blobs',
            '3d_lowres_sparse_sparsepatches40p',
            '3d_lowres_sparse_hybridsparsepatchesslices40p',
            '3d_lowres_sparse_pixelwise10',
            '3d_lowres_sparse_pixelwise5',
            '3d_lowres_sparse_pixelwise30',
            '3d_lowres_sparse_randOrthSlices3',
            '3d_lowres_sparse_randOrthSlices5',
            '3d_lowres_sparse_rand_ori_slices_with_oversampling_10',
            '3d_lowres_sparse_rand_ori_slices_with_oversampling_5',
            '3d_lowres_sparse_rand_ori_slices_with_oversampling_3',
        ),
        # 994: (
        #     "3d_fullres",
        #     "3d_fullres_sparse_slicewise10",
        #     "3d_fullres_sparse_slicewise30",
        #     "3d_fullres_sparse_randblobs",
        #     "3d_fullres_sparse_pixelwise",
        #     '3d_fullres_sparse_hybridsparsepatchesslices',
        #     '3d_fullres_sparse_sparsepatches',
        #     '3d_fullres_sparse_sliceOSfg10',
        #     '3d_fullres_sparse_slicewiserand10',
        #     '3d_fullres_sparse_blobs'
        # ),
    }

    datasets = (216,)
    for d in datasets:
        all_results_file = join(nnUNet_results, f'sparse_annotationm_evaluation_{d}.csv')
        collect_results(use_these_trainers, [d], all_results_file, configurations_all[d], folds=(0, ))

    # low number of train cases
    configurations_all = {
        216: (
            "3d_lowres",
        ),
        994: (
            "3d_fullres",
        ),
    }
    for d in datasets:
        all_results_file = join(nnUNet_results, f'sparse_annotationm_evaluation_lowpercTrainCases_{d}.csv')
        collect_results(use_these_trainers, [d], all_results_file, configurations_all[d], folds=tuple(np.arange(5, 25)))