import numpy as np
from acvl_utils.morphology.morphology_helper import generate_ball

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class SparseSegBlobsPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose):
        super().__init__(verbose)
        self.num_foreground_spheres_per_class = 1 # if self.num_foreground_spheres_per_class is <1 we use it as probability per class!
        self.labeled_fraction = 0.03
        self.num_blobs = 20

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        seg = seg[0]
        label_manager = plans_manager.get_label_manager(dataset_json)

        median_image_size = configuration_manager.median_image_size_in_voxels
        spacing = configuration_manager.spacing

        if len(median_image_size) == 2:
            try:
                fullres_config = plans_manager.get_configuration('3d_fullres')
                median_image_size = fullres_config.median_image_size_in_voxels
                spacing = fullres_config.spacing
            except KeyError:
                raise RuntimeError('This preprocessor does not work on 2D datasets')

        median_image_volume = np.prod(median_image_size, dtype=np.int64)
        sphere_volume_pixels = self.labeled_fraction / self.num_blobs * median_image_volume
        vol_per_pixel = np.prod(spacing)
        sphere_volume = vol_per_pixel * sphere_volume_pixels

        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"

        labeled_pixels = 0
        allowed_labeled_pixels = self.labeled_fraction * np.prod(seg.shape, dtype=np.int64)

        # print(
        #     f'shape: {seg.shape}, allowed_labeled_pixels: {allowed_labeled_pixels}, est num spheres '
        #     f'{allowed_labeled_pixels / sphere_volume_pixels}')

        final_mask = np.zeros_like(seg, dtype=bool)
        num_spheres = 0
        locs = DefaultPreprocessor._sample_foreground_locations(seg,
                                                                label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions,
                                                                seed=None, verbose=False)
        keys = [i for i in list(locs.keys()) if len(locs[i]) > 0]
        for c in keys:
            if c != 0:
                if self.num_foreground_spheres_per_class >= 1:
                    for n in range(self.num_foreground_spheres_per_class):
                        l = [int(i) for i in locs[c].astype(float)[np.random.choice(len(locs[c]))]]
                        sphere_radius = (sphere_volume * 3 / 4 / np.pi) ** (1 / 3)
                        b = generate_ball([sphere_radius] * 3, spacing, dtype=bool)
                        x = max(0, l[0] - b.shape[0] // 2)
                        y = max(0, l[1] - b.shape[1] // 2)
                        z = max(0, l[2] - b.shape[2] // 2)
                        x = min(seg.shape[0] - b.shape[0], x)
                        y = min(seg.shape[1] - b.shape[1], y)
                        z = min(seg.shape[2] - b.shape[2], z)
                        final_mask[x:x + b.shape[0], y:y + b.shape[1], z:z + b.shape[2]][b] = True
                        added_pixels = np.sum(b)
                        labeled_pixels += added_pixels
                        num_spheres += 1
                else:
                    if np.random.uniform() < self.num_foreground_spheres_per_class:
                        # code duplication yummy. It's gotta be quick and dirty. Sorry bois.
                        l = [int(i) for i in locs[c].astype(float)[np.random.choice(len(locs[c]))]]
                        sphere_radius = (sphere_volume * 3 / 4 / np.pi) ** (1 / 3)
                        b = generate_ball([sphere_radius] * 3, spacing, dtype=bool)
                        x = max(0, l[0] - b.shape[0] // 2)
                        y = max(0, l[1] - b.shape[1] // 2)
                        z = max(0, l[2] - b.shape[2] // 2)
                        x = min(seg.shape[0] - b.shape[0], x)
                        y = min(seg.shape[1] - b.shape[1], y)
                        z = min(seg.shape[2] - b.shape[2], z)
                        final_mask[x:x + b.shape[0], y:y + b.shape[1], z:z + b.shape[2]][b] = True
                        added_pixels = np.sum(b)
                        labeled_pixels += added_pixels
                        num_spheres += 1

        while True: # guarantees at least one random sphere
            sphere_radius = (sphere_volume * 3 / 4 / np.pi) ** (1 / 3)
            b = generate_ball([sphere_radius] * 3, spacing, dtype=bool)
            # figure out if we can add this ball
            added_pixels = np.sum(b)
            current_percentage = labeled_pixels / allowed_labeled_pixels
            theoretical_next = (labeled_pixels + added_pixels) / allowed_labeled_pixels
            if np.abs(current_percentage - 1) > np.abs(theoretical_next - 1):
                x = np.random.randint(0, seg.shape[0] - b.shape[0]) if b.shape[0] != seg.shape[0] else 0
                y = np.random.randint(0, seg.shape[1] - b.shape[1]) if b.shape[1] != seg.shape[1] else 0
                z = np.random.randint(0, seg.shape[2] - b.shape[2]) if b.shape[2] != seg.shape[2] else 0
                final_mask[x:x + b.shape[0], y:y + b.shape[1], z:z + b.shape[2]][b] = True
                labeled_pixels += added_pixels
                num_spheres += 1
            else:
                break
        # print(num_spheres)
        ret = np.ones_like(seg) * label_manager.ignore_label
        ret[final_mask] = seg[final_mask]
        return ret[None]


class SparseSegBlobsPreprocessor5(SparseSegBlobsPreprocessor):
    def __init__(self, verbose):
        super().__init__(verbose)
        self.num_foreground_spheres_per_class = 1 / 57 * 5  # do not ask. You wouldn't understand


class SparseSegBlobsPreprocessor10(SparseSegBlobsPreprocessor):
    def __init__(self, verbose):
        super().__init__(verbose)
        self.num_foreground_spheres_per_class = 1 / 57 * 10  # do not ask. You wouldn't understand


class SparseSegBlobsPreprocessor30(SparseSegBlobsPreprocessor):
    def __init__(self, verbose):
        super().__init__(verbose)
        self.num_foreground_spheres_per_class = 1 / 57 * 30  # do not ask. You wouldn't understand


class SparseSegBlobsPreprocessor50(SparseSegBlobsPreprocessor):
    def __init__(self, verbose):
        super().__init__(verbose)
        self.num_foreground_spheres_per_class = 1 / 57 * 50  # do not ask. You wouldn't understand


class SparseSegRandomBlobsPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.labeled_fraction = 0.03
        self.num_blobs = 20

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        seg = seg[0]
        label_manager = plans_manager.get_label_manager(dataset_json)

        median_image_size = configuration_manager.median_image_size_in_voxels
        spacing = configuration_manager.spacing

        if len(median_image_size) == 2:
            try:
                fullres_config = plans_manager.get_configuration('3d_fullres')
                median_image_size = fullres_config.median_image_size_in_voxels
                spacing = fullres_config.spacing
            except KeyError:
                raise RuntimeError('This preprocessor does not work on 2D datasets')

        median_image_volume = np.prod(median_image_size, dtype=np.int64)
        sphere_volume_pixels = self.labeled_fraction / self.num_blobs * median_image_volume
        vol_per_pixel = np.prod(spacing)
        sphere_volume = vol_per_pixel * sphere_volume_pixels

        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"

        labeled_pixels = 0
        allowed_labeled_pixels = self.labeled_fraction * np.prod(seg.shape, dtype=np.int64)

        print(
            f'shape: {seg.shape}, allowed_labeled_pixels: {allowed_labeled_pixels}, est num spheres '
            f'{allowed_labeled_pixels / sphere_volume_pixels}')

        final_mask = np.zeros_like(seg, dtype=bool)
        num_spheres = 0
        while True:
            sphere_radius = (sphere_volume * 3 / 4 / np.pi) ** (1 / 3)
            b = generate_ball([sphere_radius] * 3, spacing, dtype=bool)
            # figure out if we can add this ball
            added_pixels = np.sum(b)
            current_percentage = labeled_pixels / allowed_labeled_pixels
            theoretical_next = (labeled_pixels + added_pixels) / allowed_labeled_pixels
            if np.abs(current_percentage - 1) > np.abs(theoretical_next - 1):
                x = np.random.randint(0, seg.shape[0] - b.shape[0]) if b.shape[0] != seg.shape[0] else 0
                y = np.random.randint(0, seg.shape[1] - b.shape[1]) if b.shape[1] != seg.shape[1] else 0
                z = np.random.randint(0, seg.shape[2] - b.shape[2]) if b.shape[2] != seg.shape[2] else 0
                final_mask[x:x + b.shape[0], y:y + b.shape[1], z:z + b.shape[2]][b] = True
                labeled_pixels += added_pixels
                num_spheres += 1
            else:
                break
        print(num_spheres)
        ret = np.ones_like(seg) * label_manager.ignore_label
        ret[final_mask] = seg[final_mask]
        return ret[None]


class SparseSegRandomBlobsPreprocessor5(SparseSegRandomBlobsPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.labeled_fraction = 0.05


class SparseSegRandomBlobsPreprocessor10(SparseSegRandomBlobsPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.labeled_fraction = 0.1


class SparseSegRandomBlobsPreprocessor30(SparseSegRandomBlobsPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.labeled_fraction = 0.3


class SparseSegRandomBlobsPreprocessor50(SparseSegRandomBlobsPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.labeled_fraction = 0.5


