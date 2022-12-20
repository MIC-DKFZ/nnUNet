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
                        num_patches_taken += 1

        # sample random slices until targeted_annotated_pixels_percent is met
        current_percent_pixels = np.sum(seg_new != label_manager.ignore_label) / np.prod(seg.shape, dtype=np.int64)
        diff = self.targeted_annotated_pixels_percent - current_percent_pixels
        percent_pixels_per_axis_cutoffs = current_percent_pixels + diff / 3, current_percent_pixels + 2 / 3 * diff

        current_percent_pixels = percent_pixels_per_axis_cutoffs[0] - 1e-8  # guarantee at least one slice
        while current_percent_pixels < percent_pixels_per_axis_cutoffs[0]:
            s = np.random.choice(seg.shape[0])
            seg_new[s] = seg[s]
            current_percent_pixels = np.sum(seg_new != label_manager.ignore_label) / np.prod(seg.shape, dtype=np.int64)
        current_percent_pixels = percent_pixels_per_axis_cutoffs[0] - 1e-8  # guarantee at least one slice
        while current_percent_pixels < percent_pixels_per_axis_cutoffs[1]:
            s = np.random.choice(seg.shape[1])
            seg_new[:, s] = seg[:, s]
            current_percent_pixels = np.sum(seg_new != label_manager.ignore_label) / np.prod(seg.shape, dtype=np.int64)
        current_percent_pixels = percent_pixels_per_axis_cutoffs[0] - 1e-8  # guarantee at least one slice
        while current_percent_pixels < self.targeted_annotated_pixels_percent:
            s = np.random.choice(seg.shape[2])
            seg_new[:, :, s] = seg[:, :, s]
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


class SparsePatchesAndSlicesPreprocessor3_3(SparsePatchesAndSlicesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)
        self.targeted_annotated_pixels_percent = 0.03
        self.patches_per_class = 0.3


class SparsePatchesAndSlicesPreprocessor10_2(SparsePatchesAndSlicesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)
        self.targeted_annotated_pixels_percent = 0.085
        self.patches_per_class = 0.15


class SparsePatchesAndSlicesPreprocessor3_4(SparsePatchesAndSlicesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)
        self.targeted_annotated_pixels_percent = 0.03
        self.patches_per_class = 0.2


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



class SparsePatchesPreprocessor3(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)  # for amos (142, 240, 240) this makes (142*240**2)/(48**3*0.2)=370 patches for a fully annotated image (respecting patch_annotation_density_per_dim)
        self.patch_annotation_density_per_dim = 0.067
        self.patches_per_class = 0.2 # if this is <1 then this is interpreted as probability of taking a patch
        self.num_random_patches = 10

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # one patch per class, the rest random. Patches are sparsely annotated
        seg = seg[0]
        label_manager = plans_manager.get_label_manager(dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        # patch size should follow image aspect ratio
        seg_new = np.ones_like(seg) * label_manager.ignore_label

        locs = DefaultPreprocessor._sample_foreground_locations(
            seg,
            label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions,
            seed=None, verbose=False)

        # pick a random patch per class
        for c in locs.keys():
            if len(locs[c]) > 0:
                if self.patches_per_class > 1:
                    for p in range(self.patches_per_class):
                        x, y, z = locs[c].astype(float)[np.random.choice(len(locs[c]))]
                        x, y, z = int(x - self.patch_size[0] // 2), int(y - self.patch_size[1] // 2), int(z - self.patch_size[2] // 2)
                        x = max(0, x)
                        y = max(0, y)
                        z = max(0, y)
                        x = min(seg.shape[0] - self.patch_size[0], x)
                        y = min(seg.shape[1] - self.patch_size[1], y)
                        z = min(seg.shape[2] - self.patch_size[2], z)
                        slicer = (slice(x, x + self.patch_size[0]), slice(y, y + self.patch_size[1]), slice(z, z + self.patch_size[2]))
                        # not best practice lol
                        ret = SparseSegSliceRandomOrth.modify_seg_fn(self, seg[slicer][None],
                                                                     plans_manager, dataset_json, configuration_manager,
                                                                     self.patch_annotation_density_per_dim)[0]
                        seg_new[slicer] = ret
                else:
                    if np.random.uniform() < self.patches_per_class:
                        x, y, z = locs[c].astype(float)[np.random.choice(len(locs[c]))]
                        x, y, z = int(x - self.patch_size[0] // 2), int(y - self.patch_size[1] // 2), int(z - self.patch_size[2] // 2)
                        x = max(0, x)
                        y = max(0, y)
                        z = max(0, y)
                        x = min(seg.shape[0] - self.patch_size[0], x)
                        y = min(seg.shape[1] - self.patch_size[1], y)
                        z = min(seg.shape[2] - self.patch_size[2], z)
                        slicer = (slice(x, x + self.patch_size[0]), slice(y, y + self.patch_size[1]), slice(z, z + self.patch_size[2]))
                        # not best practice lol
                        ret = SparseSegSliceRandomOrth.modify_seg_fn(self, seg[slicer][None],
                                                                     plans_manager, dataset_json, configuration_manager,
                                                                     self.patch_annotation_density_per_dim)[0]
                        seg_new[slicer] = ret

        for i in range(self.num_random_patches):
            # pick a random location, verify that there is no to little overlap with existing patches
            x = np.random.choice(seg.shape[0] - self.patch_size[0])
            y = np.random.choice(seg.shape[1] - self.patch_size[1])
            z = np.random.choice(seg.shape[2] - self.patch_size[2])
            slicer = (slice(x, x + self.patch_size[0]), slice(y, y + self.patch_size[1]), slice(z, z + self.patch_size[2]))

            ret = SparseSegSliceRandomOrth.modify_seg_fn(self, seg[slicer][None], plans_manager, dataset_json,
                                                         configuration_manager,
                                                         self.patch_annotation_density_per_dim)[0]
            seg_new[slicer] = ret
        return seg_new[None]


class SparsePatchesPreprocessor3_2(SparsePatchesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48,
                           48)  # for amos (142, 240, 240) this makes (142*240**2)/(48**3*0.2)=370 patches for a fully annotated image (respecting patch_annotation_density_per_dim)
        self.patch_annotation_density_per_dim = 0.067
        self.num_random_patches = 10
        self.patches_per_class = 0.3  # if this is <1 then this is interpreted as probability of taking a patch


class SparsePatchesPreprocessor3_3(SparsePatchesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48,
                           48)  # for amos (142, 240, 240) this makes (142*240**2)/(48**3*0.2)=370 patches for a fully annotated image (respecting patch_annotation_density_per_dim)
        self.patch_annotation_density_per_dim = 0.067
        self.num_random_patches = 10
        self.patches_per_class = 0.5  # if this is <1 then this is interpreted as probability of taking a patch


class SparsePatchesPreprocessor10_2(SparsePatchesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48, 48)
        self.patch_annotation_density_per_dim = 0.067
        self.num_random_patches = 30
        self.patches_per_class = 0.2  # if this is <1 then this is interpreted as probability of taking a patch


class SparsePatchesPreprocessor5(SparsePatchesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (48, 48,
                           48)  # for amos (142, 240, 240) this makes (142*240**2)/(48**3*0.2)=370 patches for a fully annotated image (respecting patch_annotation_density_per_dim)
        self.patch_annotation_density_per_dim = 0.067
        self.num_random_patches = 20
        self.patches_per_class = 0.3  # if this is <1 then this is interpreted as probability of taking a patch


class SparsePatchesPreprocessor10(SparsePatchesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (64, 64,
                           64)  # for amos (142, 240, 240) this makes (142*240**2)/(48**3*0.2)=370 patches for a fully annotated image (respecting patch_annotation_density_per_dim)
        self.patch_annotation_density_per_dim = 0.1
        self.num_random_patches = 20
        self.patches_per_class = 0.5  # if this is <1 then this is interpreted as probability of taking a patch


class SparsePatchesPreprocessor30(SparsePatchesPreprocessor3):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.patch_size = (64, 64,
                           64)  # for amos (142, 240, 240) this makes (142*240**2)/(48**3*0.2)=370 patches for a fully annotated image (respecting patch_annotation_density_per_dim)
        self.patch_annotation_density_per_dim = 0.15
        self.num_random_patches = 20
        self.patches_per_class = 1  # if this is <1 then this is interpreted as probability of taking a patch

