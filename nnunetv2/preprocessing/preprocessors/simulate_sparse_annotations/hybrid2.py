# sparse patches
import numpy as np
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.preprocessors.simulate_sparse_annotations.slices import SparseSegSliceRandomOrth
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager



class SparsePatchesAndSlicesPreprocessor3(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)  # for amos (142, 240, 240) this makes (142*240**2)/(48**3*0.2)=370 patches for a fully annotated image (respecting patch_annotation_density_per_dim)
        self.patch_annotation_density_per_dim = 0.067
        self.targeted_annotated_pixels_percent = 0.03
        self.patches_per_class = 1 # if this is <1 then this is interpreted as probability of taking a patch

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # one patch per class, the rest random. Patches are sparsely annotated
        seg = seg[0]
        label_manager = plans_manager.get_label_manager(dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        # patch size should follow image aspect ratio

        num_patches_taken = 0
        patch_mask = np.zeros_like(seg, dtype=bool)
        seg_new = np.ones_like(seg) * label_manager.ignore_label

        locs = DefaultPreprocessor._sample_foreground_locations(
            seg,
            label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions,
            seed=None, verbose=False)
        # pick a random patch per class
        for c in locs.keys():
            if len(locs[c]) > 0:
                if self.patches_per_class > 1:
                    for i in range(self.patches_per_class):
                        x, y, z = locs[c][np.random.choice(len(locs[c]))]
                        x, y, z = int(x - self.patch_size[0] // 2), int(y - self.patch_size[1] // 2), int(z - self.patch_size[2] // 2)
                        x = max(0, x)
                        y = max(0, y)
                        z = max(0, y)
                        x = min(seg.shape[0] - self.patch_size[0], x)
                        y = min(seg.shape[1] - self.patch_size[1], y)
                        z = min(seg.shape[2] - self.patch_size[2], z)
                        slicer = (slice(x, x + self.patch_size[0]), slice(y, y + self.patch_size[1]), slice(z, z + self.patch_size[2]))
                        # not best practice lol
                        ret = \
                            SparseSegSliceRandomOrth.modify_seg_fn(self, seg[slicer][None], plans_manager, dataset_json,
                                                                   configuration_manager,
                                                                   self.patch_annotation_density_per_dim)[0]
                        seg_new[slicer] = ret
                        patch_mask[slicer] = True
                        num_patches_taken += 1
                else:
                    if np.random.uniform() < self.patches_per_class:
                        x, y, z = locs[c][np.random.choice(len(locs[c]))]
                        x, y, z = int(x - self.patch_size[0] // 2), int(y - self.patch_size[1] // 2), int(z - self.patch_size[2] // 2)
                        x = max(0, x)
                        y = max(0, y)
                        z = max(0, y)
                        x = min(seg.shape[0] - self.patch_size[0], x)
                        y = min(seg.shape[1] - self.patch_size[1], y)
                        z = min(seg.shape[2] - self.patch_size[2], z)
                        slicer = (slice(x, x + self.patch_size[0]), slice(y, y + self.patch_size[1]), slice(z, z + self.patch_size[2]))
                        # not best practice lol
                        ret = \
                            SparseSegSliceRandomOrth.modify_seg_fn(self, seg[slicer][None], plans_manager, dataset_json,
                                                                   configuration_manager,
                                                                   self.patch_annotation_density_per_dim)[0]
                        seg_new[slicer] = ret
                        patch_mask[slicer] = True
                        num_patches_taken += 1

        # sample random slices until targeted_annotated_pixels_percent is met
        current_percent_pixels = np.sum(seg_new != label_manager.ignore_label) / np.prod(seg.shape, dtype=np.int64)
        diff = self.targeted_annotated_pixels_percent - current_percent_pixels
        assert diff > 0
        percent_pixels_per_axis_cutoffs = current_percent_pixels + diff / 3, current_percent_pixels + 2 / 3 * diff

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


class SparsePatchesAndSlicesPreprocessor5(SparsePatchesAndSlicesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)
        self.targeted_annotated_pixels_percent = 0.05
        self.patches_per_class = 1


class SparsePatchesAndSlicesPreprocessor3_2(SparsePatchesAndSlicesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)
        self.targeted_annotated_pixels_percent = 0.03
        self.patches_per_class = 0.5


class SparsePatchesAndSlicesPreprocessor10(SparsePatchesAndSlicesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)
        self.targeted_annotated_pixels_percent = 0.1
        self.patches_per_class = 1


class SparsePatchesAndSlicesPreprocessor30(SparsePatchesAndSlicesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (64, 64, 64)
        self.targeted_annotated_pixels_percent = 0.3
        self.patch_annotation_density_per_dim = 0.1
        self.patches_per_class = 1


class SparsePatchesAndSlicesPreprocessor50(SparsePatchesAndSlicesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (64, 64, 64)
        self.targeted_annotated_pixels_percent = 0.5
        self.patch_annotation_density_per_dim = 0.2
        self.patches_per_class = 1

