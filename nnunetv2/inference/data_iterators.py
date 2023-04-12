from typing import Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists: List[List[str]],
                 list_of_segs_from_prev_stage_files: Union[None, List[str]],
                 preprocessor: DefaultPreprocessor,
                 output_filenames_truncated: Union[None, List[str]],
                 plans_manager: PlansManager,
                 dataset_json: dict,
                 configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1):
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json = \
            preprocessor, plans_manager, configuration_manager, dataset_json

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage_files is None:
            list_of_segs_from_prev_stage_files = [None] * len(list_of_lists)
        if output_filenames_truncated is None:
            output_filenames_truncated = [None] * len(list_of_lists)

        super().__init__(list(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)),
                         1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        files = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        ofile = self._data[idx][2]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg, data_properites = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
                                                                self.configuration_manager,
                                                                self.dataset_json)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data)

        return {'data': data, 'data_properites': data_properites, 'ofile': ofile}


class PreprocessAdapterFromNpy(DataLoader):
    def __init__(self, list_of_images: List[np.ndarray],
                 list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                 list_of_image_properties: List[dict],
                 truncated_ofnames: Union[List[str], None],
                 plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1, verbose: bool = False):
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json, self.truncated_ofnames = \
            preprocessor, plans_manager, configuration_manager, dataset_json, truncated_ofnames

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage is None:
            list_of_segs_from_prev_stage = [None] * len(list_of_images)
        if truncated_ofnames is None:
            truncated_ofnames = [None] * len(list_of_images)

        super().__init__(
            list(zip(list_of_images, list_of_segs_from_prev_stage, list_of_image_properties, truncated_ofnames)),
            1, num_threads_in_multithreaded,
            seed_for_shuffle=1, return_incomplete=True,
            shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_images)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        image = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        props = self._data[idx][2]
        ofname = self._data[idx][3]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg, data_properites = self.preprocessor.run_case_npy(image, seg_prev_stage, props,
                                                                    self.plans_manager,
                                                                    self.configuration_manager,
                                                                    self.dataset_json)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data)

        return {'data': data, 'data_properites': data_properites, 'ofile': ofname}


def get_data_iterator_from_lists_of_filenames(input_list_of_lists: List[List[str]],
                                              seg_from_prev_stage_files: Union[List[str], None],
                                              output_filenames_truncated: Union[List[str], None],
                                              configuration_manager: ConfigurationManager,
                                              plans_manager: PlansManager,
                                              dataset_json: dict,
                                              num_processes: int,
                                              pin_memory: bool,
                                              verbose: bool = False):
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = max(1, min(num_processes, len(input_list_of_lists)))
    ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
                            output_filenames_truncated, plans_manager, dataset_json,
                            configuration_manager, num_processes)
    if num_processes == 0:
        mta = SingleThreadedAugmenter(ppa, None)
    else:
        mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
    return mta


def get_data_iterator_from_raw_npy_data(image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        plans_manager: PlansManager,
                                        dataset_json: dict,
                                        configuration_manager: ConfigurationManager,
                                        num_processes: int = 3,
                                        pin_memory: bool = False
                                        ):
    list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
        image_or_list_of_images

    if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
        segs_from_prev_stage_or_list_of_segs_from_prev_stage = [segs_from_prev_stage_or_list_of_segs_from_prev_stage]

    if isinstance(truncated_ofname, str):
        truncated_ofname = [truncated_ofname]

    if isinstance(properties_or_list_of_properties, dict):
        properties_or_list_of_properties = [properties_or_list_of_properties]

    num_processes = min(num_processes, len(list_of_images))
    ppa = PreprocessAdapterFromNpy(list_of_images, segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                   properties_or_list_of_properties, truncated_ofname,
                                   plans_manager, dataset_json, configuration_manager, num_processes)
    if num_processes == 0:
        mta = SingleThreadedAugmenter(ppa, None)
    else:
        mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
    return mta