# sparse patches
import numpy as np
import scipy.signal
from acvl_utils.morphology.morphology_helper import generate_ball
from scipy.signal import fftconvolve

from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.preprocessors.simulate_sparse_annotations.slices import SparseSegSliceRandomOrth
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager


class SparseSparsePatchesPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.targeted_num_patches = 20  # more patches makes each patch smaller, fewer patches makes them larger
        self.patch_annotation_density_per_dim = 0.067  # x3 makes 20% annotation density
        self.targeted_annotated_pixels_percent = 0.03  # 3%

    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        # one patch per class, the rest random. Patches are sparsely annotated
        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        # patch size should follow image aspect ratio
        pixels_in_patches_percent = self.targeted_annotated_pixels_percent / (self.patch_annotation_density_per_dim * 3)
        patch_size = [round(i) for i in (pixels_in_patches_percent / self.targeted_num_patches) ** (1/3) * np.array(seg.shape)]

        num_patches_taken = 0
        patch_mask = np.zeros_like(seg, dtype=bool)
        seg_new = np.ones_like(seg) * label_manager.ignore_label

        locs = DefaultPreprocessor._sample_foreground_locations(seg,
                                                                label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions,
                                                                seed=None, verbose=False)
        # pick a random patch per class
        for c in locs.keys():
            if len(locs[c]) > 0:
                x, y, z = locs[c].astype(float)[np.random.choice(len(locs[c]))]
                x, y, z = int(x - patch_size[0] // 2), int(y - patch_size[1] // 2), int(z - patch_size[2] // 2)
                x = max(0, x)
                y = max(0, y)
                z = max(0, y)
                x = min(seg.shape[0] - patch_size[0], x)
                y = min(seg.shape[1] - patch_size[1], y)
                z = min(seg.shape[2] - patch_size[2], z)
                slicer = (slice(x, x + patch_size[0]), slice(y, y + patch_size[1]), slice(z, z + patch_size[2]))
                # not best practice lol
                ret = SparseSegSliceRandomOrth.modify_seg_fn(self, seg[slicer][None], plans, dataset_json, configuration, self.patch_annotation_density_per_dim)[0]
                seg_new[slicer] = ret
                patch_mask[slicer] = True
                num_patches_taken += 1

        # random patches are random, with no overlap to existing patches

        # we should have at least one background patch!
        if num_patches_taken <= self.targeted_num_patches:
            targeted_num_patches = num_patches_taken + 1
        else:
            targeted_num_patches = self.targeted_num_patches

        allowed_overlap_percent = 0.1
        while num_patches_taken < targeted_num_patches:
            # pick a random location, verify that there is no to little overlap with existing patches
            x = np.random.choice(seg.shape[0] - patch_size[0])
            y = np.random.choice(seg.shape[1] - patch_size[1])
            z = np.random.choice(seg.shape[2] - patch_size[2])
            slicer = (slice(x, x + patch_size[0]), slice(y, y + patch_size[1]), slice(z, z + patch_size[2]))

            if np.sum(patch_mask[slicer]) > allowed_overlap_percent * np.prod(patch_size):
                # too much overlap with existing patches
                continue

            ret = SparseSegSliceRandomOrth.modify_seg_fn(self, seg[slicer][None], plans, dataset_json, configuration, self.patch_annotation_density_per_dim)[0]
            seg_new[slicer] = ret
            patch_mask[slicer] = True
            num_patches_taken += 1
        return seg_new[None]


class SparseSparsePatchesPreprocessor40P(SparseSparsePatchesPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.targeted_num_patches = 40  # more patches makes each patch smaller, fewer patches makes them larger


class SparseHybridSparsePatchesSlicesPreprocessor(SparseSparsePatchesPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.targeted_num_patches = 20  # comes from SparseSparsePatchesPreprocessor. The total amount of patches the
        # number of classes, not this. We still need this here to control the patch size though.

    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        # one patch per class, the rest random. Patches are sparsely annotated
        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        # patch size should follow image aspect ratio
        pixels_in_patches_percent = self.targeted_annotated_pixels_percent / (self.patch_annotation_density_per_dim * 3)
        patch_size = [round(i) for i in
                      (pixels_in_patches_percent / self.targeted_num_patches) ** (1 / 3) * np.array(seg.shape)]

        num_patches_taken = 0
        patch_mask = np.zeros_like(seg, dtype=bool)
        seg_new = np.ones_like(seg) * label_manager.ignore_label

        locs = DefaultPreprocessor._sample_foreground_locations(seg,
                                                                label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions,
                                                                seed=None, verbose=False)
        # pick a random patch per class
        for c in locs.keys():
            if len(locs[c]) > 0:
                x, y, z = locs[c].astype(float)[np.random.choice(len(locs[c]))]
                x, y, z = int(x - patch_size[0] // 2), int(y - patch_size[1] // 2), int(z - patch_size[2] // 2)
                x = max(0, x)
                y = max(0, y)
                z = max(0, y)
                x = min(seg.shape[0] - patch_size[0], x)
                y = min(seg.shape[1] - patch_size[1], y)
                z = min(seg.shape[2] - patch_size[2], z)
                slicer = (slice(x, x + patch_size[0]), slice(y, y + patch_size[1]), slice(z, z + patch_size[2]))
                # not best practice lol
                ret = \
                SparseSegSliceRandomOrth.modify_seg_fn(self, seg[slicer][None], plans, dataset_json, configuration,
                                                       self.patch_annotation_density_per_dim)[0]
                seg_new[slicer] = ret
                patch_mask[slicer] = True
                num_patches_taken += 1

        # sample random slices until targeted_annotated_pixels_percent is met
        current_percent_pixels = np.sum(seg_new != label_manager.ignore_label) / np.prod(seg.shape, dtype=np.int64)
        diff = self.targeted_annotated_pixels_percent - current_percent_pixels
        assert diff > 0
        percent_pixels_per_axis_cutoffs = current_percent_pixels + diff / 3, current_percent_pixels + 2/3 * diff

        current_percent_pixels = percent_pixels_per_axis_cutoffs[0] - 1e-8  # guarantee at least one slice
        while current_percent_pixels < percent_pixels_per_axis_cutoffs[0]:
            s = np.random.choice(seg.shape[0])
            seg_new[s] = seg[s]
            patch_mask[s] = True
            current_percent_pixels = np.sum(seg_new != label_manager.ignore_label) / np.prod(seg.shape, dtype=np.int64)
        current_percent_pixels = percent_pixels_per_axis_cutoffs[0] - 1e-8  # guarantee at least one slice
        while current_percent_pixels < percent_pixels_per_axis_cutoffs[1]:
            s = np.random.choice(seg.shape[1])
            seg_new[:, s] = seg[:, s]
            patch_mask[:, s] = True
            current_percent_pixels = np.sum(seg_new != label_manager.ignore_label) / np.prod(seg.shape, dtype=np.int64)
        current_percent_pixels = percent_pixels_per_axis_cutoffs[0] - 1e-8  # guarantee at least one slice
        while current_percent_pixels < self.targeted_annotated_pixels_percent:
            s = np.random.choice(seg.shape[2])
            seg_new[:, :, s] = seg[:, :, s]
            patch_mask[:, :, s] = True
            current_percent_pixels = np.sum(seg_new != label_manager.ignore_label) / np.prod(seg.shape, dtype=np.int64)

        return seg_new[None]


class SparseHybridSparsePatchesSlicesPreprocessor40P(SparseHybridSparsePatchesSlicesPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.targeted_num_patches = 40  # comes from SparseSparsePatchesPreprocessor. The total amount of patches the
        # number of classes, not this. We still need this here to control the patch size though.
