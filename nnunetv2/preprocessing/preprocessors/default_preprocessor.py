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
from multiprocessing import Pool
from typing import Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

import nnunetv2
from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnunetv2.paths import default_plans_identifier
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_nf_by_name
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.utils import get_caseIDs_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder, extract_unique_classes_from_dataset_json_labels


class DefaultPreprocessor(object):
    def __init__(self):
        """
        Everything we need is in the plans. Those are given when run() is called

        CAREFUL! WE USE INT8 FOR SAVING SEGMENTATIONS (NOT UINT8) SO 127 IS THE MAXIMUM LABEL!
        """

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans: Union[dict, str], configuration_name: str,
                 dataset_json: Union[dict, str], dataset_fingerprint: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(plans, str):
            plans = load_json(plans)
        if isinstance(dataset_fingerprint, str):
            dataset_fingerprint = load_json(dataset_fingerprint)
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        configuration = plans['configurations'][configuration_name]
        rw = recursive_find_reader_writer_by_name(plans['image_reader_writer'])()

        # load image(s)
        data, data_properites = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans['transpose_forward']]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans['transpose_forward']]])
        original_spacing = [data_properites['spacing'][i] for i in plans['transpose_forward']]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        data_properites['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        print(data.shape, seg.shape)
        data_properites['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        fn_data = recursive_find_resampling_nf_by_name(configuration['resampling_fn_data'])
        fn_seg = recursive_find_resampling_nf_by_name(configuration['resampling_fn_seg'])
        target_spacing = configuration['spacing']  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 3d we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        data = fn_data(data, new_shape, original_spacing, target_spacing,
                       **configuration['resampling_fn_data_kwargs'])
        seg = fn_seg(seg, new_shape, original_spacing, target_spacing,
                     **configuration['resampling_fn_seg_kwargs'])

        # normalize
        data = self._normalize(data, seg, configuration['normalization_schemes'],
                               dataset_fingerprint)

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if seg_file is not None:
            classes = [j for i, j in dataset_json['labels'].items() if i != 'background']
            data_properites['class_locations'] = self._sample_foreground_locations(seg, classes)

        return data, seg.astype(np.int8), data_properites

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans: Union[dict, str], configuration_name: str, dataset_json: Union[dict, str],
                      dataset_fingerprint: Union[dict, str]):
        data, seg, properties = self.run_case(image_files, seg_file, plans, configuration_name, dataset_json,
                                              dataset_fingerprint)
        # print('dtypes', data.dtype, seg.dtype)
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes: List[int], seed: int = 1234):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes:
            if isinstance(c, tuple):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
            # print(c, target_num_samples)
        return class_locs

    def _normalize(self, data: np.ndarray, seg: np.ndarray, normalization_schemes: List[str],
                   dataset_fingerprint: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnunetv2.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError('Unable to locate class \'%s\' for normalization' % scheme)
            normalizer = normalizer_class(normalization_schemes,
                                          dataset_fingerprint['foreground_intensity_properties_by_modality'][str(c)])
            data[c] = normalizer.run(data, seg)
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str, num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file

        plans = load_json(plans_file)
        print(f'Preprocessing the following configuration: {configuration_name}')
        print(plans['configurations'][configuration_name])

        if configuration_name not in plans['configurations'].keys():
            raise RuntimeError("Requested configuration '%s' not found in ")

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)
        classes = extract_unique_classes_from_dataset_json_labels(dataset_json['labels'])
        if max(classes) > 127:
            raise RuntimeError('WE USE INT8 FOR SAVING SEGMENTATIONS (NOT UINT8) SO 127 IS THE MAXIMUM LABEL! '
                               'Your labels go larger than that')

        dataset_fingerprint = load_json(join(nnUNet_preprocessed, dataset_name, 'dataset_fingerprint.json'))

        caseids = get_caseIDs_from_splitted_dataset_folder(join(nnUNet_raw, dataset_name, 'imagesTr'),
                                                           dataset_json['file_ending'])
        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_name)
        maybe_mkdir_p(output_directory)

        output_filenames_truncated = [join(output_directory, i) for i in caseids]

        suffix = dataset_json['file_ending']
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(join(nnUNet_raw, dataset_name, 'imagesTr'), suffix,
                                                                 caseids)
        # list of segmentation filenames
        seg_fnames = [join(nnUNet_raw, dataset_name, 'labelsTr', i + suffix) for i in caseids]

        pool = Pool(num_processes)
        # we submit the datasets one by one so that we don't have dangling processes in the end
        # self.run_case_save(*list(zip(output_filenames_truncated, image_fnames, seg_fnames))[0], plans,
        #                   configuration_name)
        # raise RuntimeError()
        results = []
        for ofname, ifnames, segfnames in zip(output_filenames_truncated, image_fnames, seg_fnames):
            results.append(pool.starmap_async(self.run_case_save,
                                              ((ofname, ifnames, segfnames, plans, configuration_name,
                                                dataset_json, dataset_fingerprint),)))
        # let the workers do their job
        [i.get() for i in results]


def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = default_plans_identifier + '.json'
    dataset_json_file = 'dataset.json'
    dataset_fingerprint_file = 'dataset_fingerprint.json'
    input_images = ['case000_0000.nii.gz']  # if you only have one modality, you still need a list: ['case000_0000.nii.gz']

    configuration = '3d_fullres'
    pp = DefaultPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    data, _, properties = pp.run_case(input_images, seg_file=None, plans=plans_file, configuration_name=configuration,
                                      dataset_json=dataset_json_file, dataset_fingerprint=dataset_fingerprint_file)

    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return data


if __name__ == '__main__':
    pp = DefaultPreprocessor()
    pp.run(2, '2d', default_plans_identifier, 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()


