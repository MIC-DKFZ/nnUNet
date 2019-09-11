from copy import deepcopy
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subdirs
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props_poolLateV2
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.plan_and_preprocess_task import create_lists_from_splitted_dataset, crop
from nnunet.experiment_planning.configuration import FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK, \
    batch_size_covers_max_percent_of_dataset, dataset_min_batch_size_cap, RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD, \
    HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0
import shutil
from nnunet.paths import *
from nnunet.network_architecture.generic_UNet import Generic_UNet


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
        self.plans_fname = join(self.preprocessed_output_folder, default_plans_identifier + "isoPatchesInVoxels_plans_3D.pkl")

    def plan_experiment(self):
        architecture_input_voxels = np.prod(Generic_UNet.DEFAULT_PATCH_SIZE_3D)

        def get_stage(target_spacing, original_spacing, original_shape, num_cases,
                              num_modalities, num_classes):
            """
            we start with new_median_shape and then work our way towards isotropic patches. if the patches are
            isotropic, remove from the axis with the largest spacing (example: 128x128x128 but spacing is (3, 1, 1)
            then remove from first axis)
            """
            new_median_shape = np.round(original_spacing / target_spacing * original_shape).astype(int)
            dataset_num_voxels = np.prod(new_median_shape) * num_cases

            input_patch_size = new_median_shape

            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp,  \
                shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(input_patch_size,
                                                                     FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                                     Generic_UNet.MAX_NUMPOOL_3D, target_spacing)

            ref = Generic_UNet.use_this_for_batch_size_computation_3D
            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                Generic_UNet.BASE_NUM_FEATURES_3D,
                                                                Generic_UNet.MAX_NUM_FILTERS_3D, num_modalities,
                                                                num_classes,
                                                                pool_op_kernel_sizes)
            while here > ref:
                # find the largest axis. If patch is isotropic, pick the axis with the largest spacing
                if len(np.unique(new_shp)) == 1:
                    largest_axis = np.argsort(target_spacing)[-1]
                else:
                    largest_axis = np.argsort(new_shp)[-1]

                # now let's recompute things for the new shape
                tmp = deepcopy(new_shp)
                tmp[largest_axis] -= shape_must_be_divisible_by[largest_axis]
                _, _, _, _, shape_must_be_divisible_by_new = get_pool_and_conv_props_poolLateV2(tmp,
                                                                     FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                                     Generic_UNet.MAX_NUMPOOL_3D, target_spacing)
                # that is because we want to subtract the shape_must_be_divisible_by_new, not shape_must_be_divisible_by
                new_shp[largest_axis] -= shape_must_be_divisible_by_new[largest_axis]

                # we have to recompute everything for real now:
                network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
                shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(new_shp,
                                                                     FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK,
                                                                     Generic_UNet.MAX_NUMPOOL_3D, target_spacing)

                here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                    Generic_UNet.BASE_NUM_FEATURES_3D,
                                                                    Generic_UNet.MAX_NUM_FILTERS_3D, num_modalities,
                                                                    num_classes, pool_op_kernel_sizes)

            input_patch_size = new_shp

            batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D  # This is what works with DEFAULT_PATCH_SIZE_3D
            batch_size = int(np.floor(max(ref / here, 1) * batch_size))

            # check if batch size is too large
            max_batch_size = np.round(batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                      np.prod(input_patch_size)).astype(int)
            max_batch_size = max(max_batch_size, dataset_min_batch_size_cap)
            batch_size = min(batch_size, max_batch_size)

            do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[0]) > RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD

            plan = {
                'batch_size': batch_size,
                'num_pool_per_axis': network_num_pool_per_axis,
                'patch_size': input_patch_size,
                'median_patient_size_in_voxels': new_median_shape,
                'current_spacing': target_spacing,
                'original_spacing': original_spacing,
                'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
                'pool_op_kernel_sizes': pool_op_kernel_sizes,
                'conv_kernel_sizes': conv_kernel_sizes,
            }
            return plan

        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero maks for normalizaion?", use_nonzero_mask_for_normalization)
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK, " in the bottleneck")

        # fullres stage
        self.plans_per_stage = list()

        self.plans_per_stage.append(get_stage(target_spacing, target_spacing, median_shape,
                                                             len(self.list_of_cropped_npz_files),
                                                             num_modalities, len(all_classes) + 1))

        # check if we need lowres
        # thanks Zakiyi (https://github.com/MIC-DKFZ/nnUNet/issues/61) for spotting this bug :-)
        #if np.prod(self.plans_per_stage[-1]['median_patient_size_in_voxels'], dtype=np.int64) / \
        #        architecture_input_voxels < HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0:
        if np.prod(self.plans_per_stage[-1]['median_patient_size_in_voxels'], dtype=np.int64) / \
                np.prod(self.plans_per_stage[-1]['patch_size'], dtype=np.int64) < HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0:
            more = False
        else:
            more = True

        if more:
            #print("now onto lowres")
            # this is kind of a chicken egg problem because we don't know what patch size a network can process on 12GB
            # without knowing the target spacing but we cannot know that target spacing without knowing what patch
            # size a network can process because we want the patch size to be about the median shape of the downsampled
            # cases. We therefore do it the dumb way and reduce the target spacing iteratively
            lowres_stage_spacing = deepcopy(target_spacing)
            num_voxels = np.prod(median_shape)

            while num_voxels > HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0 * architecture_input_voxels:
                max_spacing = max(lowres_stage_spacing)
                if np.any((max_spacing / lowres_stage_spacing) > 2):
                    lowres_stage_spacing[(max_spacing / lowres_stage_spacing) > 2] \
                        *= 1.01
                else:
                    lowres_stage_spacing *= 1.01
                #print(lowres_stage_spacing)

                num_voxels = np.prod(target_spacing / lowres_stage_spacing * median_shape)

                new = get_stage(lowres_stage_spacing, target_spacing, median_shape, len(self.list_of_cropped_npz_files),
                                num_modalities, len(all_classes) + 1)
                architecture_input_voxels = np.prod(new['patch_size'])

            if 2 * np.prod(new['median_patient_size_in_voxels']) < np.prod(self.plans_per_stage[0]['median_patient_size_in_voxels']):
                self.plans_per_stage.append(new)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        print(self.plans_per_stage)

        normalization_schemes = self.determine_normalization_scheme()
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = \
            self.determine_postprocessing()

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_output_folder, 'num_classes': len(all_classes),
                 'all_classes': all_classes, 'base_num_features': Generic_UNet.BASE_NUM_FEATURES_3D,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'keep_only_largest_region': only_keep_largest_connected_component,
                 'min_region_size_per_class': min_region_size_per_class, 'min_size_per_class': min_size_per_class,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage}

        self.plans = plans
        self.save_my_plans()


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

    tasks = []
    for i in task_ids:
        i = int(i)
        candidates = subdirs(cropped_output_dir, prefix="Task%02.0d" % i, join=False)
        assert len(candidates) == 1
        tasks.append(candidates[0])

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

