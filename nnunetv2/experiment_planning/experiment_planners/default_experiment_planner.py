from copy import deepcopy
from typing import Type, List, Union, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_json, write_json, isdir, join, save_json
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch.nn.modules.conv import _ConvNd

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization
from nnunetv2.preprocessing.normalization.map_modality_to_normalization import get_normalization_scheme
from nnunetv2.utilities.utils import get_caseIDs_from_splitted_dataset_folder, create_lists_from_splitted_dataset_folder
import numpy as np
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_spacing, \
    resample_data_or_seg_to_shape, compute_new_shape
from nnunetv2.configuration import ANISO_THRESHOLD


class ExperimentPlanner(object):
    def __init__(self, raw_data_folder: str, dataset_json_file: str, preprocessed_output_folder: str,
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'GenericPreprocessor', plans_name: str = 'nnUNetPlans',
                 data_identifier: str = 'nnUNetData',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 overwrite_modalities: Union[List[str], Tuple[str, ...]] = None, ):
        self.dataset_json = load_json(dataset_json_file)
        self.raw_data_folder = raw_data_folder
        self.preprocessed_output_folder = preprocessed_output_folder

        # load dataset fingerprint
        self.dataset_properties = load_json(join(self.raw_data_folder, 'dataset_properties.json'))

        self.anisotropy_threshold = ANISO_THRESHOLD

        self.UNet_base_num_features = 32
        self.UNet_class = PlainConvUNet
        self.UNet_reference_val = 9999  # todo
        self.UNet_reference_com_nfeatures = 30  # todo legacy stuff
        self.UNet_reference_val_corresp_GB = 8
        self.UNet_reference_val_corresp_bs = 2
        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_featuremap_min_edge_length = 4
        self.UNet_blocks_per_stage = 2
        self.UNet_min_batch_size = 2
        self.UNet_max_features_2d = 512
        self.UNet_max_features_3d = 320

        self.lowres_creation_threshold = 0.25  # if the patch size of fullres is less than 25% of the voxels in the
        # median shape then we need a lowres config as well

        self.preprocessor_name = preprocessor_name
        self.plans_name = plans_name
        self.data_identifier = data_identifier
        self.overwrite_target_spacing = overwrite_target_spacing \
 \
        # can be used to overwrite normalization scheme
        if overwrite_modalities is not None:
            assert len(overwrite_modalities) == len(self.dataset_json['modalities'])
        self.overwrite_modalities = overwrite_modalities

        self.plans = None

    def determine_reader_writer(self):
        training_identifiers = get_caseIDs_from_splitted_dataset_folder(join(self.raw_data_folder, 'imagesTr'),
                                                                             self.dataset_json['file_ending'])
        return determine_reader_writer(self.dataset_json, join(self.raw_data_folder, 'imagesTr',
                                                               training_identifiers[0] + '_0000' +
                                                               self.dataset_json['file_ending']))

    def estimate_VRAM_usage(self, patch_size,
                            n_stages: int,
                            strides: Union[int, List[int], Tuple[int, ...]]):
        """
        Works for PlainConvUNet, ResidualEncoderUNet
        """
        dim = len(patch_size)
        conv_op = convert_dim_to_conv_op(dim)
        norm_op = get_matching_instancenorm(conv_op)
        max_features = self.UNet_max_features_2d if len(patch_size) == 2 else self.UNet_max_features_3d
        net = self.UNet_class(len(self.dataset_json['file_ending']['modalities'].keys()), n_stages,
                              [min(max_features, self.UNet_reference_com_nfeatures * 2 ** i) for i in range(n_stages)],
                              conv_op, 3, strides, self.UNet_blocks_per_stage,
                              len(self.dataset_json['file_ending']['labels'].keys()),
                              self.UNet_blocks_per_stage[::-1] if not isinstance(self.UNet_blocks_per_stage, int)
                              else self.UNet_blocks_per_stage,
                              norm_op=norm_op)
        return net.compute_conv_feature_map_size(patch_size)

    def determine_resampling(self):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)
        """
        resampling_data = resample_data_or_seg_to_spacing
        resampling_data_kwargs = {
            "is_seg": False,
            "order": 3,
            "order_z": 0,
            "force_separate_z": None,
        }
        resampling_seg = resample_data_or_seg_to_spacing
        resampling_seg_kwargs = {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow
        """
        resampling_fn = resample_data_or_seg_to_shape
        resampling_fn_kwargs = {
            "is_seg": True,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
        return resampling_fn, resampling_fn_kwargs

    def determine_fullres_target_spacing(self) -> np.ndarray:
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        if self.overwrite_target_spacing is not None:
            return np.array(self.overwrite_target_spacing)

        spacings = self.dataset_properties['spacings']
        sizes = self.dataset_properties['shapes_after_crop']

        target = np.percentile(np.vstack(spacings), 50, 0)

        # todo sizes_after_resampling = [compute_new_shape(j, i, target) for i, j in zip(spacings, sizes)]

        target_size = np.percentile(np.vstack(sizes), 50, 0)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    def determine_normalization_scheme(self) -> List[Type[ImageNormalization]]:
        modalities = self.dataset_json['modalities'] if self.overwrite_modalities is None else self.overwrite_modalities
        normalization_schemes = [get_normalization_scheme(m) for m in self.dataset_json['modalities']]
        return normalization_schemes

    def determine_whether_to_use_mask_for_norm(self) -> bool:
        # use the nonzero mask for normalization if the cropping resulted in a substantial decrease in
        # image size (this is an indication that the data is something like brats/isles and then we want to
        # normalize in the brain region only)

        # remember that the normalization scheme decides whether or not this is actually used! Only ZScoreNormalization
        # can use it as of now. The others ignore it

        use_nonzero_mask_for_norm = self.dataset_properties['median_relative_size_after_cropping'] < (3 / 4.)

        return use_nonzero_mask_for_norm

    def determine_transpose(self):
        # todo we should use shapes for that as well
        target_spacing = self.determine_fullres_target_spacing()

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        transpose_forward = [max_spacing_axis] + remaining_axes
        transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
        return transpose_forward, transpose_backward

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...], List[int]],
                                    num_training_cases: int):
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]

        # clip initial patch size to median_shape. It makes little sense to have it be larger than that
        initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])])

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)

        # now estimate vram consumption
        estimate = self.estimate_VRAM_usage(patch_size, len(pool_op_kernel_sizes) + 1, pool_op_kernel_sizes)

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = self.UNet_reference_val / self.UNet_reference_val_corresp_GB * self.UNet_vram_target_GB
        # adapt for our min batch size
        reference = reference / self.UNet_reference_val_corresp_bs * self.UNet_min_batch_size

        while estimate > reference:
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort(patch_size / median_shape[:len(spacing)])[-1]

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                 self.UNet_featuremap_min_edge_length,
                                                                 999999)
            estimate = self.estimate_VRAM_usage(patch_size, len(pool_op_kernel_sizes) + 1, pool_op_kernel_sizes)

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        batch_size = round((reference / estimate) * self.UNet_min_batch_size)

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        approximate_n_voxels_dataset = np.prod(median_shape, dtype=np.float64) * num_training_cases
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * 0.05 / np.prod(patch_size, dtype=np.float64))
        batch_size = max(max(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        do_dummy_2D_data_aug = (max(patch_size) / patch_size[0]) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': patch_size,
            'median_patient_size_in_voxels': median_shape,
            'spacing': spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'unet_max_num_features': self.UNet_max_features_3d if len(spacing) == 3 else self.UNet_max_features_2d
        }
        return plan

    def plan_experiment(self):
        # first get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing.transpose(transpose_forward)

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_properties['spacings'], self.dataset_properties['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape.transpose(transpose_forward)

        # only run 3d if this is a 3d dataset
        if new_median_shape_transposed.shape[0] != 1:
            plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                               new_median_shape_transposed,
                                                               self.dataset_json['numTraining'])
            # maybe add 3d_lowres as well
            patch_size_fullres = plan_3d_fullres['patch_size']
            median_num_voxels = np.prod(new_median_shape_transposed, dtype=np.float64)
            num_voxels_in_patch = np.prod(patch_size_fullres, dtype=np.float64)

            plan_3d_lowres = None
            lowres_spacing = deepcopy(plan_3d_fullres['spacing'])

            while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:
                # we incrementally increase the target spacing. We start with the anisotropic axis/axes until it/they
                # is/are similar (factor 2) to the other ax(i/e)s.
                max_spacing = max(lowres_spacing)
                if np.any((max_spacing / lowres_spacing) > 2):
                    lowres_spacing[(max_spacing / lowres_spacing) > 2] *= 1.01
                else:
                    lowres_spacing *= 1.01
                median_num_voxels = np.prod(plan_3d_fullres['spacing'] / lowres_spacing * new_median_shape_transposed,
                                            dtype=np.float64)
                plan_3d_lowres = self.get_plans_for_configuration(lowres_spacing,
                                                                  plan_3d_fullres['spacing'] / lowres_spacing *
                                                                  new_median_shape_transposed,
                                                                  self.dataset_json['numTraining'])
                num_voxels_in_patch = np.prod(plan_3d_lowres['patch_size'], dtype=np.int64)
        else:
            plan_3d_fullres = None
            plan_3d_lowres = None

        # 2D configuration
        plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
                                                   new_median_shape_transposed,
                                                   self.dataset_json['numTraining'])

        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_properties['spacings'], 0).transpose(transpose_forward)
        median_shape = np.median(self.dataset_properties['shapes_after_crop'], 0).transpose(transpose_forward)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_fn_softmax, resampling_fn_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        plans = {'dataset_properties': self.dataset_properties,
                 'transpose_forward': transpose_forward,
                 'transpose_backward': transpose_backward,
                 'original_median_spacing_after_transp': median_spacing,
                 'original_median_shape_after_transp': median_shape,
                 'normalization_schemes': self.determine_normalization_scheme(),
                 'UNet_base_num_features': self.UNet_base_num_features,
                 'data_identifier': self.data_identifier,
                 'preprocessor_name': self.preprocessor_name,
                 'use_mask_for_norm': self.determine_whether_to_use_mask_for_norm(),
                 'dataset_json': self.dataset_json,
                 'image_reader_writer': self.determine_reader_writer(),
                 'UNet_class_name': self.UNet_class.__name,
                 'resampling_fn_data': resampling_data.__name__,
                 'resampling_fn_seg': resampling_seg.__name__,
                 'resampling_fn_data_kwargs': resampling_data_kwargs,
                 'resampling_fn_seg_kwargs': resampling_seg_kwargs,
                 'softmax_resample_fn': resampling_fn_softmax.__name__,
                 'softmax_resample_fn_kwargs': resampling_fn_softmax_kwargs,
                 '2d': plan_2d}

        if plan_3d_fullres is not None:
            plans['3d_fullres'] = plan_3d_fullres
        if plan_3d_lowres is not None:
            plans['3d_lowres'] = plan_3d_lowres

        save_json(plans, join(self.preprocessed_output_folder, self.plans_name + '.json'))
        self.plans = plans

    def load_plans(self, fname: str):
        self.plans = load_json(fname)

    def run_preprocessing(self, do_2d: bool = True, do_3d_fullres: bool = True, do_3d_lowres: bool = True,
                          n_processes_2D: int = 8, n_processes_3d_fullres: int = 6, n_processes_3d_lowres: int = 8):
        assert self.plans is not None, 'self.plans is none. Either load a plans file with self.load_plans or run ' \
                                       'self.plan_experiment'
