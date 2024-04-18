import os
import warnings
from typing import List, Union, Type

import numpy as np
import blosc2
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.utils import unpack_dataset
import math


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
            data = np.load(join(self.source_folder, identifier + '.npz'))['data']
        else:
            data = np.load(data_npy_file, mmap_mode='r')

        seg_npy_file = join(self.source_folder, identifier + '_seg.npy')
        if not isfile(seg_npy_file):
            seg = np.load(join(self.source_folder, identifier + '.npz'))['seg']
        else:
            seg = np.load(seg_npy_file, mmap_mode='r')

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_npy_file = join(self.folder_with_segs_from_previous_stage, identifier + '.npy')
            if isfile(prev_seg_npy_file):
                seg_prev = np.load(prev_seg_npy_file, 'r')
            else:
                seg_prev = np.load(join(self.folder_with_segs_from_previous_stage, identifier + '.npz'))['seg']
            seg = np.vstack((seg, seg_prev[None]))

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
    

class nnUNetDatasetBlosc2(object):
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
        data_b2nd_file = join(self.source_folder, identifier + '.b2nd')
        data = blosc2.open(urlpath=data_b2nd_file, mode='r')

        seg_b2nd_file = join(self.source_folder, identifier + '_seg.b2nd')
        seg = blosc2.open(urlpath=seg_b2nd_file, mode='r')

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_b2nd_file = join(self.folder_with_segs_from_previous_stage, identifier + '.b2nd')
            seg_prev = blosc2.open(urlpath=prev_seg_b2nd_file, mode='r')
            seg = np.vstack((seg, seg_prev[None]))

        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        return data, seg, properties

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str,
            chunks = None,
            blocks = None,
            chunks_seg = None,
            blocks_seg = None
    ):
        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks
        if blocks is not None and blocks[0] <= data.shape[0] and blocks[1] <= data.shape[1] and blocks[2] <= data.shape[2]:
            blosc2.asarray(data, urlpath=output_filename_truncated + '.b2nd', chunks=chunks, blocks=blocks)
            blosc2.asarray(seg, urlpath=output_filename_truncated + '_seg.b2nd', chunks=chunks_seg, blocks=blocks_seg)
            write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str,
            chunks_seg = None,
            blocks_seg = None
    ):
        blosc2.asarray(seg, urlpath=output_filename_truncated + '_seg.b2nd', chunks=chunks_seg, blocks=blocks_seg)

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-5] for i in os.listdir(folder) if i.endswith("b2nd")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify_npy: bool = True):
        pass

    @staticmethod
    def comp_blosc2_params(patch_size, num_channels, data_bit_size, l1_cache_size=32768, l3_cache_size=1441792):  # 11MiB / 8 = 1441792, 64MiB / 12 = 5.767e+6
        """
        Computes a recommended block and chunk size for saving arrays with blosc v2.

        Note: It is currently required that the size is the same in every dimension for the patch size!

        Args:
            patch_size: The spatial patch size without batch and channel dimensions. Can be either 2D or 3D.
            num_channels: The number of channels / modalities of the images.
            data_bit_size: The Bit size of the array. Example: float32 -> 32 Bits
            l1_cache_size: The size of the L1 cache per core in Bytes.
            l3_cache_size: The size of the L3 cache exclusively accessible by each core. Usually the global size of the L3 cache divided by the number of cores.

        Returns:
            The recommended block and the chunk size.
        """
        assert len(np.unique(patch_size)) == 1  # See note in docstring

        # Compute maximum block size candidate based on L1 cache size
        max_block_size_cand_1 = 0
        block_byte_size = 0
        i = 0
        while block_byte_size < l1_cache_size:
            i += 1
            max_block_size_cand_1 = 2**i
            block_byte_size = nnUNetDatasetBlosc2.comp_data_byte_size([max_block_size_cand_1] * len(patch_size), num_channels, data_bit_size)
        max_block_size_cand_1 = 2**(i-1)

        # Compute maximum block size candidate based on patch size
        max_block_size_cand_2 = 0
        i = 0
        while max_block_size_cand_2 < (patch_size[0] / 2):  # Block size should be smaller than the patch size. Value 2 might be changed in the future.
            i += 1
            max_block_size_cand_2 = 2**i
        max_block_size_cand_2 = 2**(i-1)

        # Compute final block size based on both candidates
        block_size = max_block_size_cand_1 if max_block_size_cand_1 < max_block_size_cand_2 else max_block_size_cand_2

        # Compute maximum recommended chunk size based on L2 cache size
        max_chunk_size_cand_1 = 0
        chunk_byte_size = 0
        i = 0
        while chunk_byte_size < l3_cache_size:
            i += 1
            max_chunk_size_cand_1 = 2**i
            chunk_byte_size = nnUNetDatasetBlosc2.comp_data_byte_size([max_chunk_size_cand_1] * len(patch_size), num_channels, data_bit_size)
        max_chunk_size_cand_1 = 2**(i-1)

        # Compute maximum block size candidate based on patch size
        max_chunk_size_cand_2 = 2**int(math.log2(block_size)+3)  # Chunk size should not be extremly larger than the block size. Value 3 might be changed in the future.

        # Compute final chunk size based on both candidates
        chunk_size = max_chunk_size_cand_1 if max_chunk_size_cand_1 < max_chunk_size_cand_2 else max_chunk_size_cand_2

        return block_size, chunk_size    

    @staticmethod
    def comp_data_byte_size(shape, num_channels, data_bit_size):
        return int(np.prod(shape)*num_channels*data_bit_size/8)


DEFAULT_DATASET_CLASS = nnUNetDatasetNumpy

file_ending_dataset_mapping = {
    'npz': nnUNetDatasetNumpy,
}


def infer_dataset_class(folder: str) -> Type[nnUNetDatasetNumpy]:
    file_endings = set([os.path.basename(i).split('.')[-1] for i in subfiles(folder, join=False)])
    if 'pkl' in file_endings:
        file_endings.remove('pkl')
    if 'npy' in file_endings:
        file_endings.remove('npy')
    assert len(file_endings) == 1, (f'Found more than one file ending in the folder {folder}. '
                                    f'Unable to infer nnUNetDataset variant!')
    return file_ending_dataset_mapping[list(file_endings)[0]]
