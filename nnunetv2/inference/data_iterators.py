import multiprocessing
import queue
from torch.multiprocessing import Event, Process, Queue, Manager

from time import sleep
from typing import Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def preprocess_fromfiles_save_to_queue(list_of_lists: List[List[str]],
                                       list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                       output_filenames_truncated: Union[None, List[str]],
                                       plans_manager: PlansManager,
                                       dataset_json: dict,
                                       configuration_manager: ConfigurationManager,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False):
    try:
        label_manager = plans_manager.get_label_manager(dataset_json)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(list_of_lists)):
            data, seg, data_properties = preprocessor.run_case(list_of_lists[idx],
                                                               list_of_segs_from_prev_stage_files[
                                                                   idx] if list_of_segs_from_prev_stage_files is not None else None,
                                                               plans_manager,
                                                               configuration_manager,
                                                               dataset_json)
            if list_of_segs_from_prev_stage_files is not None and list_of_segs_from_prev_stage_files[idx] is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {'data': data, 'data_properties': data_properties,
                    'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        # print(Exception, e)
        abort_event.set()
        raise e


def preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
                                     list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_lists), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
                     args=(
                         list_of_lists[i::num_processes],
                         list_of_segs_from_prev_stage_files[
                         i::num_processes] if list_of_segs_from_prev_stage_files is not None else None,
                         output_filenames_truncated[
                         i::num_processes] if output_filenames_truncated is not None else None,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # import IPython;IPython.embed()
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]

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
        files, seg_prev_stage, ofile = self._data[idx]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg, data_properties = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
                                                                self.configuration_manager,
                                                                self.dataset_json)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data)

        return {'data': data, 'data_properties': data_properties, 'ofile': ofile}


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
        image, seg_prev_stage, props, ofname = self._data[idx]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg = self.preprocessor.run_case_npy(image, seg_prev_stage, props,
                                                   self.plans_manager,
                                                   self.configuration_manager,
                                                   self.dataset_json)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data)

        return {'data': data, 'data_properties': props, 'ofile': ofname}


def preprocess_fromnpy_save_to_queue(list_of_images: List[np.ndarray],
                                     list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                     list_of_image_properties: List[dict],
                                     truncated_ofnames: Union[List[str], None],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     target_queue: Queue,
                                     done_event: Event,
                                     abort_event: Event,
                                     verbose: bool = False):
    try:
        label_manager = plans_manager.get_label_manager(dataset_json)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(list_of_images)):
            data, seg = preprocessor.run_case_npy(list_of_images[idx],
                                                  list_of_segs_from_prev_stage[
                                                      idx] if list_of_segs_from_prev_stage is not None else None,
                                                  list_of_image_properties[idx],
                                                  plans_manager,
                                                  configuration_manager,
                                                  dataset_json)
            if list_of_segs_from_prev_stage is not None and list_of_segs_from_prev_stage[idx] is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {'data': data, 'data_properties': list_of_image_properties[idx],
                    'ofile': truncated_ofnames[idx] if truncated_ofnames is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e


def preprocessing_iterator_fromnpy(list_of_images: List[np.ndarray],
                                   list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                   list_of_image_properties: List[dict],
                                   truncated_ofnames: Union[List[str], None],
                                   plans_manager: PlansManager,
                                   dataset_json: dict,
                                   configuration_manager: ConfigurationManager,
                                   num_processes: int,
                                   pin_memory: bool = False,
                                   verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_images), num_processes)
    assert num_processes >= 1
    target_queues = []
    processes = []
    done_events = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = manager.Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromnpy_save_to_queue,
                     args=(
                         list_of_images[i::num_processes],
                         list_of_segs_from_prev_stage[
                         i::num_processes] if list_of_segs_from_prev_stage is not None else None,
                         list_of_image_properties[i::num_processes],
                         truncated_ofnames[i::num_processes] if truncated_ofnames is not None else None,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        done_events.append(event)
        processes.append(pr)
        target_queues.append(queue)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]
