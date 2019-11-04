import shutil
from copy import deepcopy

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subdirs
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props_poolLateV2
from nnunet.experiment_planning.configuration import FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK, \
    batch_size_covers_max_percent_of_dataset, dataset_min_batch_size_cap, RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.plan_and_preprocess_task import create_lists_from_splitted_dataset, split_4d, crop
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *


class ExperimentPlanner3D_IsoPatchesInVoxels(ExperimentPlanner):
    """
    patches that are isotropic in the number of voxels (not mm), such as 128x128x128 allow more voxels to be processed
    at once because we don't have to do annoying pooling stuff

    CAREFUL!
    this one does not support transpose_forward and transpose_backward
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_IsoPatchesInVoxels, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_isoPatchesInVoxels"
        self.plans_fname = join(self.preprocessed_output_folder, default_plans_identifier + "fixedisoPatchesInVoxels_plans_3D.pkl")

    @staticmethod
    def get_properties_for_stage(current_spacing, original_spacing, original_shape, num_cases,
                                     num_modalities, num_classes):
        """
        """
        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
        dataset_num_voxels = np.prod(new_median_shape) * num_cases

        input_patch_size = new_median_shape

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(input_patch_size,
                                                                        FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                                        Generic_UNet.MAX_NUMPOOL_3D,
                                                                        current_spacing)

        ref = Generic_UNet.use_this_for_batch_size_computation_3D
        here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                            Generic_UNet.BASE_NUM_FEATURES_3D,
                                                            Generic_UNet.MAX_NUM_FILTERS_3D, num_modalities,
                                                            num_classes,
                                                            pool_op_kernel_sizes)
        while here > ref:
            # find the largest axis. If patch is isotropic, pick the axis with the largest spacing
            if len(np.unique(new_shp)) == 1:
                axis_to_be_reduced = np.argsort(current_spacing)[-1]
            else:
                axis_to_be_reduced = np.argsort(new_shp)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props_poolLateV2(tmp,
                                                   FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                   Generic_UNet.MAX_NUMPOOL_3D,
                                                   current_spacing)
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(new_shp,
                                                                            FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                                            Generic_UNet.MAX_NUMPOOL_3D,
                                                                            current_spacing)

            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                Generic_UNet.BASE_NUM_FEATURES_3D,
                                                                Generic_UNet.MAX_NUM_FILTERS_3D, num_modalities,
                                                                num_classes, pool_op_kernel_sizes)
            print(new_shp)

        input_patch_size = new_shp

        batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D  # This is what works with 128**3
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large
        max_batch_size = np.round(batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, dataset_min_batch_size_cap)
        batch_size = min(batch_size, max_batch_size)

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[
            0]) > RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        return plan


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="list of int")
    parser.add_argument("-p", action="store_true", help="set this if you actually want to run the preprocessing. If "
                                                        "this is not set then this script will only create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8, help="num_threads_lowres")
    parser.add_argument("-tf", type=int, required=False, default=8, help="num_threads_fullres")

    args = parser.parse_args()
    task_ids = args.task_ids
    run_preprocessing = args.p
    tl = args.tl
    tf = args.tf

    # we need splitted and cropped data. This could be improved in the future TODO
    tasks = []
    for i in task_ids:
        i = int(i)

        splitted_taskString_candidates = subdirs(splitted_4d_output_dir, prefix="Task%02.0d" % i, join=False)
        raw_taskString_candidates = subdirs(raw_dataset_dir, prefix="Task%02.0d" % i, join=False)
        cropped_taskString_candidates = subdirs(cropped_output_dir, prefix="Task%02.0d" % i, join=False)

        # is splitted data there?
        if len(splitted_taskString_candidates) == 0:
            # splitted not there
            assert len(raw_taskString_candidates) > 0, \
                "splitted data is not present and so is raw data (Task %d)" % i
            assert len(raw_taskString_candidates) == 1, "ambiguous task string (raw Task %d)" % i
            # split raw data into splitted
            split_4d(raw_taskString_candidates[0])
        elif len(splitted_taskString_candidates) > 1:
            raise RuntimeError("ambiguous task string (raw Task %d)" % i)
        else:
            pass

        if len(cropped_taskString_candidates) > 1:
            raise RuntimeError("ambiguous task string (raw Task %d)" % i)
        else:
            crop(splitted_taskString_candidates[0], False, tf)

        tasks.append(cropped_taskString_candidates[0])

    for t in tasks:
        try:
            print("\n\n\n", t)
            cropped_out_dir = os.path.join(cropped_output_dir, t)
            preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
            splitted_4d_output_dir_task = os.path.join(splitted_4d_output_dir, t)
            lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

            dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False)
            _ = dataset_analyzer.analyze_dataset()  # this will write output files that will be used by the ExperimentPlanner

            maybe_mkdir_p(preprocessing_output_dir_this_task)
            shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
            shutil.copy(join(splitted_4d_output_dir, t, "dataset.json"), preprocessing_output_dir_this_task)

            threads = (tl, tf)

            print("number of threads: ", threads, "\n")

            exp_planner = ExperimentPlanner3D_IsoPatchesInVoxels(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if run_preprocessing:
                exp_planner.run_preprocessing(threads)
        except Exception as e:
            print(e)

