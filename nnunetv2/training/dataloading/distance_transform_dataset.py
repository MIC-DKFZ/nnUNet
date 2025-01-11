import os
import torch
import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

class DistanceTransformDataset(nnUNetDataset):
    def load_case(self, key):
        entry = self[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
        else:
            seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        if 'open_dist_file' in entry.keys():
            dist_map = entry['open_dist_file']
        elif isfile(entry['data_file'][:-4] + "_dist.npy"):
            dist_map = np.load(entry['data_file'][:-4] + "_dist.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_dist_file'] = dist_map
        else:
            dist_map = np.load(entry['data_file'])['dist']

        return data, seg, dist_map, entry['properties']

    @staticmethod
    def load_nifti(file_path):
        # Implement method to load .nii.gz files
        nii = nib.load(file_path)
        return torch.tensor(nii.get_fdata(), dtype=torch.float32)
