import os
import pandas as pd
from typing import Union, Tuple, List, Optional
import numpy as np
import torch
import blosc2
import warnings
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import load_json, join, load_pickle
from threadpoolctl import threadpool_limits

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class MultiTasknnUNetDataset(nnUNetBaseDataset):
    """
    Extended nnUNet dataset for multi-task learning with classification labels
    Reads classification labels from labels.csv in preprocessed folder
    """

    def __init__(self,
                 folder: str,
                 identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None):

        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)

        # Load classification labels from CSV
        self.classification_labels = self._load_classification_labels()
        blosc2.set_nthreads(1)

    def _load_classification_labels(self) -> dict:
        """
        Load classification labels from labels.csv in preprocessed folder
        Expected format: case_id,subtype
        """
        labels_file = join('/'.join(self.source_folder.split('/')[:-1]), 'labels.csv')

        if not os.path.exists(labels_file):
            print(f"Warning: labels.csv not found at {labels_file}")
            # Return default labels (all subtype 0)
            return {case_id: 0 for case_id in self.identifiers}

        try:
            df = pd.read_csv(labels_file)
            # Ensure columns exist
            if 'case_id' not in df.columns or 'subtype' not in df.columns:
                raise ValueError("labels.csv must have 'case_id' and 'subtype' columns")

            # Create mapping
            df['case_id'] = df['case_id'].str[:8].astype(str)  # Ensure case_id is string
            labels_dict = dict(zip(df['case_id'], df['subtype']))

            # Verify all case_identifiers have labels
            missing_labels = []
            for case_id in self.identifiers:
                if case_id not in labels_dict:
                    missing_labels.append(case_id)
                    labels_dict[case_id] = 0  # Default to subtype 0

            if missing_labels:
                print(f"Warning: Missing labels for cases: {missing_labels}")

            return labels_dict

        except Exception as e:
            print(f"Error loading labels.csv: {e}")
            return {case_id: 0 for case_id in self.identifiers}

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        """Load case with classification label added"""
        dparams = {'nthreads': 1}

        # Load data and segmentation (blosc2 format)
        data_b2nd_file = join(self.source_folder, identifier + '.b2nd')
        mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}
        data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)

        seg_b2nd_file = join(self.source_folder, identifier + '_seg.b2nd')
        seg = blosc2.open(urlpath=seg_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_b2nd_file = join(self.folder_with_segs_from_previous_stage, identifier + '.b2nd')
            seg_prev = blosc2.open(urlpath=prev_seg_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)
        else:
            seg_prev = None

        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))

        # Add classification label
        classification_label = self.classification_labels.get(identifier, 0)
        properties['classification_label'] = classification_label

        return data, seg, seg_prev, properties

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str,
            chunks=None,
            blocks=None,
            chunks_seg=None,
            blocks_seg=None,
            clevel: int = 8,
            codec=blosc2.Codec.ZSTD
    ):
        """Save case in blosc2 format"""
        blosc2.set_nthreads(1)
        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks

        cparams = {
            'codec': codec,
            'clevel': clevel,
        }

        blosc2.asarray(np.ascontiguousarray(data),
                      urlpath=output_filename_truncated + '.b2nd',
                      chunks=chunks, blocks=blocks, cparams=cparams)
        blosc2.asarray(np.ascontiguousarray(seg),
                      urlpath=output_filename_truncated + '_seg.b2nd',
                      chunks=chunks_seg, blocks=blocks_seg, cparams=cparams)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(seg: np.ndarray, output_filename_truncated: str, chunks_seg=None, blocks_seg=None):
        """Save segmentation only"""
        blosc2.asarray(seg, urlpath=output_filename_truncated + '.b2nd',
                      chunks=chunks_seg, blocks=blocks_seg)

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """Get case identifiers from preprocessed folder"""
        case_identifiers = [i[:-5] for i in os.listdir(folder)
                           if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                      num_processes: int = 8, verify: bool = True):
        """Compatibility method - no unpacking needed for blosc2"""
        pass

    def get_properties_for_case(self, case_identifier: str) -> dict:
        """Get properties including classification info"""
        properties = super().get_properties_for_case(case_identifier)

        # Add classification information to properties
        if properties is None:
            properties = {}

        properties['classification_label'] = self.classification_labels.get(case_identifier, 0)
        properties['subtype'] = f"subtype{properties['classification_label']}"

        return properties

    def get_case_identifiers_with_subtypes(self) -> dict:
        """Return case identifiers grouped by subtype"""
        subtype_groups = {0: [], 1: [], 2: []}

        for case_id, label in self.classification_labels.items():
            subtype_groups[label].append(case_id)

        return subtype_groups

    def get_classification_distribution(self) -> dict:
        """Get distribution of classification labels"""
        from collections import Counter
        distribution = Counter(self.classification_labels.values())

        return {
            'subtype_0': distribution.get(0, 0),
            'subtype_1': distribution.get(1, 0),
            'subtype_2': distribution.get(2, 0),
            'total_cases': len(self.classification_labels)
        }

    def verify_dataset_integrity(self) -> dict:
        """Verify dataset has both image and segmentation files for each case"""
        missing_files = []
        subtype_issues = []

        for case_id in self.case_identifiers:
            classification_label = self.classification_labels.get(case_id, 0)
            subtype_folder = f"subtype{classification_label}"

            # Check for image file
            image_path = join(self.folder, subtype_folder, f"{case_id}_0000.nii.gz")
            seg_path = join(self.folder, subtype_folder, f"{case_id}.nii.gz")

            if not os.path.exists(image_path):
                missing_files.append(f"Missing image: {image_path}")

            if not os.path.exists(seg_path):
                missing_files.append(f"Missing segmentation: {seg_path}")

            # Verify subtype folder exists
            if not os.path.exists(join(self.folder, subtype_folder)):
                subtype_issues.append(f"Missing subtype folder: {subtype_folder}")

        return {
            'missing_files': missing_files,
            'subtype_issues': subtype_issues,
            'is_valid': len(missing_files) == 0 and len(subtype_issues) == 0,
            'classification_distribution': self.get_classification_distribution()
        }

    def _get_label_distribution(self, labels_dict: dict) -> dict:
        """Get distribution of classification labels"""
        from collections import Counter
        distribution = Counter(labels_dict.values())
        return {f'subtype_{i}': distribution.get(i, 0) for i in range(3)}

class MultiTasknnUNetDataLoader(nnUNetDataLoader):
    """
    Multi-task dataloader that returns classification targets along with segmentation data
    """

    def __init__(self,
                 data,  # MultiTasknnUNetDataset
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):

        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        # Add classification targets
        cls_all = np.zeros(self.batch_size, dtype=np.int64)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training
            force_fg = self.get_do_oversample(j)

            # Load case with classification label
            data, seg, seg_prev, properties = self._data.load_case(i)
            classification = properties['classification_label']

            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # Crop data and segmentation
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

            # Store classification target
            cls_all[j] = classification

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            return {
                'data': data_all,
                'target': seg_all,
                'classification': torch.from_numpy(cls_all),
                'keys': selected_keys
            }

        return {
            'data': data_all,
            'target': seg_all,
            'classification': cls_all,
            'keys': selected_keys
        }