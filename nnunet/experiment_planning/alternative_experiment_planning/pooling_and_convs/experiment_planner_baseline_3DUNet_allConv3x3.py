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

from copy import deepcopy
import numpy as np
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props_poolLateV2
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.plan_and_preprocess_task import create_lists_from_splitted_dataset, crop
from nnunet.preprocessing.cropping import get_case_identifier_from_npz
from nnunet.preprocessing.preprocessing import GenericPreprocessor
from nnunet.experiment_planning.configuration import FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK, TARGET_SPACING_PERCENTILE, \
    batch_size_covers_max_percent_of_dataset, dataset_min_batch_size_cap, \
    HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0, MIN_SIZE_PER_CLASS_FACTOR, \
    RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunet.paths import *
from nnunet.network_architecture.generic_UNet import Generic_UNet
from collections import OrderedDict


class ExperimentPlannerAllConv3x3(ExperimentPlanner):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlannerAllConv3x3, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.plans_fname = join(self.preprocessed_output_folder,
                                default_plans_identifier + "allConv3x3_plans_3D.pkl")

    def plan_experiment(self):
        def get_properties_for_stage(current_spacing, original_spacing, original_shape, num_cases,
                                     num_modalities, num_classes):
            """
            Computation of input patch size starts out with the new median shape (in voxels) of a dataset. This is
            opposed to prior experiments where I based it on the median size in mm. The rationale behind this is that
            for some organ of interest the acquisition method will most likely be chosen such that the field of view and
            voxel resolution go hand in hand to show the doctor what they need to see. This assumption may be violated
            for some modalities with anisotropy (cine MRI) but we will have t live with that. In future experiments I
            will try to 1) base input patch size match aspect ratio of input size in mm (instead of voxels) and 2) to
            try to enforce that we see the same 'distance' in all directions (try to maintain equal size in mm of patch)

            The patches created here attempt keep the aspect ratio of the new_median_shape

            :param current_spacing:
            :param original_spacing:
            :param original_shape:
            :param num_cases:
            :return:
            """
            new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
            dataset_num_voxels = np.prod(new_median_shape) * num_cases

            # the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
            # input_patch_size = new_median_shape

            # compute how many voxels are one mm
            input_patch_size = 1 / np.array(current_spacing)

            # normalize voxels per mm
            input_patch_size /= input_patch_size.mean()

            # create an isotropic patch of size 512x512x512mm
            input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
            input_patch_size = np.round(input_patch_size).astype(int)

            # clip it to the median shape of the dataset because patches larger then that make not much sense
            input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]

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
                argsrt = np.argsort(new_shp / new_median_shape)[::-1]
                pool_fct_per_axis = np.prod(pool_op_kernel_sizes, 0)
                bottleneck_size_per_axis = new_shp / pool_fct_per_axis
                shape_must_be_divisible_by = [shape_must_be_divisible_by[i]
                                              if bottleneck_size_per_axis[i] > 4
                                              else shape_must_be_divisible_by[i] / 2
                                              for i in range(len(bottleneck_size_per_axis))]
                new_shp[argsrt[0]] -= shape_must_be_divisible_by[argsrt[0]]

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

            batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D  # This is what wirks with 128**3
            batch_size = int(np.floor(max(ref / here, 1) * batch_size))

            # check if batch size is too large
            max_batch_size = np.round(batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                      np.prod(input_patch_size, dtype=np.int64)).astype(int)
            max_batch_size = max(max_batch_size, dataset_min_batch_size_cap)
            batch_size = min(batch_size, max_batch_size)

            do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[
                0]) > RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD

            for s in range(len(conv_kernel_sizes)):
                conv_kernel_sizes[s] = tuple(3 for _ in conv_kernel_sizes[s])

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

        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero maks for normalizaion?", use_nonzero_mask_for_normalization)
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK, " in the bottleneck")

        # how many stages will the image pyramid have?
        self.plans_per_stage = list()

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("the transposed median shape of the dataset is ", median_shape_transposed)

        self.plans_per_stage.append(get_properties_for_stage(target_spacing_transposed, target_spacing_transposed,
                                                             median_shape_transposed,
                                                             len(self.list_of_cropped_npz_files),
                                                             num_modalities, len(all_classes) + 1))

        # thanks Zakiyi (https://github.com/MIC-DKFZ/nnUNet/issues/61) for spotting this bug :-)
        # if np.prod(self.plans_per_stage[-1]['median_patient_size_in_voxels'], dtype=np.int64) / \
        #        architecture_input_voxels < HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0:
        architecture_input_voxels_here = np.prod(self.plans_per_stage[-1]['patch_size'], dtype=np.int64)
        if np.prod(self.plans_per_stage[-1]['median_patient_size_in_voxels'], dtype=np.int64) / \
                architecture_input_voxels_here < HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0:
            more = False
        else:
            more = True

        if more:
            # if we are doing more than one stage then we want the lowest stage to have exactly
            # HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0 (this is 4 by default so the number of voxels in the
            # median shape of the lowest stage must be 4 times as much as the network can process at once (128x128x128 by
            # default). Problem is that we are downsampling higher resolution axes before we start downsampling the
            # out-of-plane axis. We could probably/maybe do this analytically but I am lazy, so here
            # we do it the dumb way

            lowres_stage_spacing = deepcopy(target_spacing)
            num_voxels = np.prod(median_shape, dtype=np.int64)
            while num_voxels > HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0 * architecture_input_voxels_here:
                max_spacing = max(lowres_stage_spacing)
                if np.any((max_spacing / lowres_stage_spacing) > 2):
                    lowres_stage_spacing[(max_spacing / lowres_stage_spacing) > 2] \
                        *= 1.01
                else:
                    lowres_stage_spacing *= 1.01
                num_voxels = np.prod(target_spacing / lowres_stage_spacing * median_shape, dtype=np.int64)

                lowres_stage_spacing_transposed = np.array(lowres_stage_spacing)[self.transpose_forward]
                new = get_properties_for_stage(lowres_stage_spacing_transposed, target_spacing_transposed,
                                               median_shape_transposed,
                                               len(self.list_of_cropped_npz_files),
                                               num_modalities, len(all_classes) + 1)
                architecture_input_voxels = np.prod(new['patch_size'])

            if 1.5 * np.prod(new['median_patient_size_in_voxels'], dtype=np.int64) < np.prod(
                    self.plans_per_stage[0]['median_patient_size_in_voxels'], dtype=np.int64):
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
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage}

        self.plans = plans
        self.save_my_plans()

    def run_preprocessing(self, num_threads):
        pass


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
            _ = dataset_analyzer.analyze_dataset() # this will write output files that will be used by the ExperimentPlanner

            maybe_mkdir_p(preprocessing_output_dir_this_task)
            shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
            shutil.copy(join(splitted_4d_output_dir, t, "dataset.json"), preprocessing_output_dir_this_task)

            threads = (tl, tf)

            print("number of threads: ", threads, "\n")

            exp_planner = ExperimentPlannerAllConv3x3(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if run_preprocessing:
                exp_planner.run_preprocessing(threads)
        except Exception as e:
            print(e)

