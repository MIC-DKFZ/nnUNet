import os
import warnings
from typing import List, Union

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.utils import unpack_dataset


class nnUNetDatasetNumpy(object):
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__()
        # print('loading dataset')
        if identifiers is None:
            identifiers = self.get_identifiers(folder)
        identifiers.sort()

        self.source_folder = folder
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.identifiers = identifiers

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        data_npy_file = join(self.source_folder, identifier + '.npy')
        if not isfile(data_npy_file):
            warnings.warn(
                "Data doesn't seem to have been unpacked. This makes loading slow! Consider enabling unpacking")
            data = np.load(join(self.source_folder, identifier + '.npz'))['data']
        else:
            data = np.load(data_npy_file, mmap_mode='r')

        seg_npy_file = join(self.source_folder, identifier + '_seg.npy')
        if not isfile(seg_npy_file):
            warnings.warn(
                "Seg doesn't seem to have been unpacked. This makes loading slow! Consider enabling unpacking")
            seg = np.load(join(self.source_folder, identifier + '.npz'))['seg']
        else:
            seg = np.load(seg_npy_file, mmap_mode='r')

        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        return data, seg, properties

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', seg=seg)

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify_npy: bool = True):
        return unpack_dataset(folder, True, overwrite_existing, num_processes, verify_npy)


DEFAULT_DATASET_CLASS = nnUNetDatasetNumpy

file_ending_dataset_mapping = {
    'npz': nnUNetDatasetNumpy,
}


def infer_dataset_class(folder: str):
    file_endings = set([os.path.basename(i).split('.')[-1] for i in subfiles(folder, join=False)])
    if 'pkl' in file_endings:
        file_endings.remove('pkl')
    if 'npy' in file_endings:
        file_endings.remove('npy')
    assert len(file_endings) == 1, (f'Found more than one file ending in the folder {folder}. '
                                    f'Unable to infer nnUNetDataset variant!')
    return file_ending_dataset_mapping[list(file_endings)[0]]
