import os
from multiprocessing import Queue
from typing import Tuple, Union, List

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile

from nnunetv2.configuration import default_num_processes
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.utils import get_preprocessor_class_from_plans
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder, get_caseIDs_from_splitted_dataset_folder
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader


class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists, preprocessor, output_filename_truncated, plans, dataset_json, configuration,
                 dataset_fingerprint, num_threads_in_multithreaded=1):

        self.preprocessor, self.plans, self.configuration, self.dataset_json, self.dataset_fingerprint = \
            preprocessor, plans, configuration, dataset_json, dataset_fingerprint

        super().__init__(list(zip(list_of_lists, output_filename_truncated)), 1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

    def generate_train_batch(self):
        idx = self.get_indices()
        files, ofile = self._data[idx]
        data, _, data_properites = self.preprocessor.run_case(files, None, self.plans, self.configuration,
                                                              self.dataset_json, self.dataset_fingerprint)
        if np.prod(data.shape) > (2e9 / 4 * 0.85):
            # we need to temporarily save the preprocessed image due to process-process communication restrictions
            np.save(ofile + '.npy', data)
            data = ofile + '.npy'

        return data, data_properites, ofile


def predict_from_raw_data(list_of_lists_or_source_folder: Union[str, List[List[str]]], output_folder: str,
                          model_training_output_dir: str, use_folds: Union[Tuple[int], str] = None,
                          tile_step_size: float = 0.5, use_gaussian: bool = True,
                          use_mirroring: bool = True, perform_everything_on_gpu: bool = False,
                          verbose: bool = True, save_probabilities: bool = False, overwrite: bool = True,
                          checkpoint_name: str = 'checkpoint_final.pth',
                          num_processes_preprocessing: int = default_num_processes):
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    config = load_json(join(model_training_output_dir, 'config.json'))
    dataset_fingerprint = load_json(join(model_training_output_dir, 'dataset_fingerprint.json'))

    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])

    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]

    output_filename_truncated = [join(output_folder, i) for i in caseids]

    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        output_filename_truncated = [output_filename_truncated[i] for i in range(len(output_filename_truncated)) if not tmp[i]]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in
                                          range(len(list_of_lists_or_source_folder)) if not tmp[i]]
        caseids = [caseids[i] for i in range(len(caseids)) if not tmp[i]]

    # we need to somehow get the configuration. We could do this via the path but I think this is not ideal. Maybe we
    # need to save an extra file?
    preprocessor = get_preprocessor_class_from_plans(plans, config['configuration_name'])()

    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = min(num_processes_preprocessing, len(list_of_lists_or_source_folder))
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, preprocessor, output_filename_truncated, plans,
                            dataset_json, config['configuration_name'], dataset_fingerprint, num_processes)
    mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None , None)

    # load parameters

    # go go go
    for data, data_properites, ofile_truncated in mta:
        

