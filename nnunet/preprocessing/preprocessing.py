#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

from collections import OrderedDict
from copy import deepcopy

from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.configuration import default_num_threads, RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD
from nnunet.preprocessing.cropping import get_case_identifier_from_npz, ImageCropper
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing.pool import Pool


def get_do_separate_z(spacing, anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0, force_separate_z=False,
                     order_z_data=0, order_z_seg=0,
                     separate_z_anisotropy_threshold=RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD):
    """
    :param data:
    :param seg:
    :param original_spacing:
    :param target_spacing:
    :param order_data:
    :param order_seg:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately
    :param order_z_seg: only applies if do_separate_z is True
    :param order_z_data: only applies if do_separate_z is True
    :param separate_z_anisotropy_threshold: if max_spacing > separate_z_anisotropy_threshold * min_spacing (per axis)
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"
    if seg is not None:
        assert len(seg.shape) == 4, "seg must be c x y z"

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(data, new_shape, False, axis, order_data, do_separate_z,
                                             order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(seg, new_shape, True, axis, order_seg, do_separate_z, order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == len(data.shape) - 1
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None].astype(dtype_data))
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None].astype(dtype_data))
                else:
                    reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data


class GenericPreprocessor(object):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list), intensityproperties=None):
        """

        :param normalization_scheme_per_modality: dict {0:'nonCT'}
        :param use_nonzero_mask: {0:False}
        :param intensityproperties:
        """
        self.transpose_forward = transpose_forward
        self.intensityproperties = intensityproperties
        self.normalization_scheme_per_modality = normalization_scheme_per_modality
        self.use_nonzero_mask = use_nonzero_mask

        self.resample_separate_z_anisotropy_threshold = RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD
        self.resample_order_data = 3
        self.resample_order_seg = 1

    @staticmethod
    def load_cropped(cropped_output_dir, case_identifier):
        all_data = np.load(os.path.join(cropped_output_dir, "%s.npz" % case_identifier))['data']
        data = all_data[:-1].astype(np.float32)
        seg = all_data[-1:]
        with open(os.path.join(cropped_output_dir, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return data, seg, properties

    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0

        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing,
                                     self.resample_order_data, self.resample_order_seg,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities" + f"{len(self.normalization_scheme_per_modality)}:{len(data)}"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                print('no intensity normalization')
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                    data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                    data[c][mask == 0] = 0
                else:
                    mn = data[c].mean()
                    std = data[c].std()
                    # print(data[c].shape, data[c].dtype, mn, std)
                    data[c] = (data[c] - mn) / (std + 1e-8)
        return data, seg, properties

    def preprocess_test_case(self, data_files, target_spacing, seg_file=None, force_separate_z=None):
        data, seg, properties = ImageCropper.crop_from_list_of_files(data_files, seg_file)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(data, target_spacing, properties, seg,
                                                            force_separate_z=force_separate_z)
        return data.astype(np.float32), seg, properties

    def _run_internal(self, target_spacing, case_identifier, output_folder_stage, cropped_output_dir, force_separate_z,
                      all_classes):
        data, seg, properties = self.load_cropped(cropped_output_dir, case_identifier)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(data, target_spacing,
                                                            properties, seg, force_separate_z)

        all_data = np.vstack((data, seg)).astype(np.float32)

        # we need to find out where the classes are and sample some random locations
        # let's do 10.000 samples per class
        # seed this for reproducibility!
        num_samples = 10000
        min_percent_coverage = 0.01 # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
        rndst = np.random.RandomState(1234)
        class_locs = {}
        for c in all_classes:
            all_locs = np.argwhere(all_data[-1] == c)
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
            print(c, target_num_samples)
        properties['class_locations'] = class_locs

        print("saving: ", os.path.join(output_folder_stage, "%s.npz" % case_identifier))
        np.savez_compressed(os.path.join(output_folder_stage, "%s.npz" % case_identifier),
                            data=all_data.astype(np.float32))
        with open(os.path.join(output_folder_stage, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        """

        :param target_spacings: list of lists [[1.25, 1.25, 5]]
        :param input_folder_with_cropped_npz: dim: c, x, y, z | npz_file['data'] np.savez_compressed(fname.npz, data=arr)
        :param output_folder:
        :param num_threads:
        :param force_separate_z: None
        :return:
        """
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        maybe_mkdir_p(output_folder)
        num_stages = len(target_spacings)
        if not isinstance(num_threads, (list, tuple, np.ndarray)):
            num_threads = [num_threads] * num_stages

        assert len(num_threads) == num_stages

        # we need to know which classes are present in this dataset so that we can precompute where these classes are
        # located. This is needed for oversampling foreground
        all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']

        for i in range(num_stages):
            all_args = []
            output_folder_stage = os.path.join(output_folder, data_identifier + "_stage%d" % i)
            maybe_mkdir_p(output_folder_stage)
            spacing = target_spacings[i]
            for j, case in enumerate(list_of_cropped_npz_files):
                case_identifier = get_case_identifier_from_npz(case)
                args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes
                all_args.append(args)
            p = Pool(num_threads[i])
            p.starmap(self._run_internal, all_args)
            p.close()
            p.join()


class GenericPreprocessor_linearResampling(GenericPreprocessor):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list),
                 intensityproperties=None):
        super().__init__(normalization_scheme_per_modality, use_nonzero_mask, transpose_forward, intensityproperties)
        self.resample_order_data = 1
        self.resample_order_seg = 1


class Preprocessor3DDifferentResampling(GenericPreprocessor):
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0

        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=3, order_z_seg=1,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        return data, seg, properties


class Preprocessor3DBetterResampling(GenericPreprocessor):
    """
    This preprocessor always uses force_separate_z=False. It does resampling to the target spacing with third
    order spline for data (just like GenericPreprocessor) and seg (unlike GenericPreprocessor). It never does separate
    resampling in z.
    """
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=False):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """
        if force_separate_z is not False:
            print("WARNING: Preprocessor3DBetterResampling always uses force_separate_z=False. "
                  "You specified %s. Your choice is overwritten" % str(force_separate_z))
            force_separate_z = False

        # be safe
        assert force_separate_z is False

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0

        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 3,
                                     force_separate_z=force_separate_z, order_z_data=99999, order_z_seg=99999,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        return data, seg, properties


class PreprocessorFor2D(GenericPreprocessor):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list), intensityproperties=None):
        super(PreprocessorFor2D, self).__init__(normalization_scheme_per_modality, use_nonzero_mask,
                                                transpose_forward, intensityproperties)

    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        assert len(list_of_cropped_npz_files) != 0, "set list of files first"
        maybe_mkdir_p(output_folder)
        all_args = []
        num_stages = len(target_spacings)

        # we need to know which classes are present in this dataset so that we can precompute where these classes are
        # located. This is needed for oversampling foreground
        all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']

        for i in range(num_stages):
            output_folder_stage = os.path.join(output_folder, data_identifier + "_stage%d" % i)
            maybe_mkdir_p(output_folder_stage)
            spacing = target_spacings[i]
            for j, case in enumerate(list_of_cropped_npz_files):
                case_identifier = get_case_identifier_from_npz(case)
                args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes
                all_args.append(args)
        p = Pool(num_threads)
        p.starmap(self._run_internal, all_args)
        p.close()
        p.join()

    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        target_spacing[0] = original_spacing_transposed[0]
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) >= len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) >= len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        print("normalization...")

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        print("normalization done")
        return data, seg, properties


class PreprocessorFor2D_edgeLength512(PreprocessorFor2D):
    target_edge_size = 512

    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        data_shape = data.shape[-2:]
        smaller_edge = min(data_shape)
        target_edge_size = self.target_edge_size
        scale_factor = target_edge_size / smaller_edge
        new_shape = [1] + [int(np.round(i * scale_factor)) for i in data_shape]
        print(new_shape)

        data = resample_data_or_seg(data, new_shape, False, None, 3, False, 0)
        seg = resample_data_or_seg(seg, new_shape, True, None, 1, False, 0)

        after = {
            'spacing': 'None',
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        print("normalization...")

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        print("normalization done")
        return data, seg, properties


class PreprocessorFor2D_edgeLength768(PreprocessorFor2D_edgeLength512):
    target_edge_size = 768


class PreprocessorFor3D_LeaveOriginalZSpacing(GenericPreprocessor):
    """
    3d_lowres and 3d_fullres are not resampled along z!
    """
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        if target_spacing[0] is None or nan we use original_spacing_transposed[0] (no resampling along z)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0
        target_spacing = deepcopy(target_spacing)
        if target_spacing[0] is None or np.isnan(target_spacing[0]):
            target_spacing[0] = original_spacing_transposed[0]
        #print(target_spacing, original_spacing_transposed)
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        st = "before:" + str(before) + '\nafter' + str(after) + "\n"
        print(st)

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        return data, seg, properties

    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        for i in range(len(target_spacings)):
            target_spacings[i][0] = None
        super().run(target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
                    default_num_threads, force_separate_z)


class PreprocessorFor3D_NoResampling(GenericPreprocessor):
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        if target_spacing[0] is None or nan we use original_spacing_transposed[0] (no resampling along z)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0
        target_spacing = deepcopy(original_spacing_transposed)
        #print(target_spacing, original_spacing_transposed)
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        st = "before:" + str(before) + '\nafter' + str(after) + "\n"
        print(st)

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        return data, seg, properties

