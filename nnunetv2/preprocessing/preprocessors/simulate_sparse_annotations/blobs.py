import numpy as np
from acvl_utils.morphology.morphology_helper import generate_ball

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager


class SparseSegBlobsPreprocessor(DefaultPreprocessor):
    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        labeled_fraction = 0.03
        num_foreground_spheres_per_class = 1

        median_patient_size = plans['configurations'][configuration]["median_patient_size_in_voxels"]
        spacing = plans['configurations'][configuration]["spacing"]

        if len(median_patient_size) == 2:
            try:
                median_patient_size = plans['configurations']['3d_fullres']["median_patient_size_in_voxels"]
                spacing = plans['configurations']['3d_fullres']["spacing"]
            except KeyError:
                raise RuntimeError('This preprocessor does not work on 2D datasets')

        median_patient_volume = np.prod(median_patient_size, dtype=np.int64)
        sphere_volume_pixels = labeled_fraction / 20 * median_patient_volume
        vol_per_pixel = np.prod(spacing)
        sphere_volume = vol_per_pixel * sphere_volume_pixels

        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"

        labeled_pixels = 0
        allowed_labeled_pixels = labeled_fraction * np.prod(seg.shape, dtype=np.int64)

        print(
            f'shape: {seg.shape}, allowed_labeled_pixels: {allowed_labeled_pixels}, est num spheres '
            f'{allowed_labeled_pixels / sphere_volume_pixels}')

        final_mask = np.zeros_like(seg, dtype=bool)
        num_spheres = 0
        locs = DefaultPreprocessor._sample_foreground_locations(seg,
                                                                label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions,
                                                                seed=None, verbose=False)
        keys = [i for i in list(locs.keys()) if len(locs[i]) > 0]
        for c in keys:
            if c != 0:
                for n in range(num_foreground_spheres_per_class):
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
        print(num_spheres)
        ret = np.ones_like(seg) * label_manager.ignore_label
        ret[final_mask] = seg[final_mask]
        return ret[None]


class SparseSegRandomBlobsPreprocessor(DefaultPreprocessor):
    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        labeled_fraction = 0.03

        median_patient_size = plans['configurations'][configuration]["median_patient_size_in_voxels"]
        spacing = plans['configurations'][configuration]["spacing"]

        if len(median_patient_size) == 2:
            try:
                median_patient_size = plans['configurations']['3d_fullres']["median_patient_size_in_voxels"]
                spacing = plans['configurations']['3d_fullres']["spacing"]
            except KeyError:
                raise RuntimeError('This preprocessor does not work on 2D datasets')

        median_patient_volume = np.prod(median_patient_size, dtype=np.int64)
        sphere_volume_pixels = labeled_fraction / 20 * median_patient_volume
        vol_per_pixel = np.prod(spacing)
        sphere_volume = vol_per_pixel * sphere_volume_pixels

        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"

        labeled_pixels = 0
        allowed_labeled_pixels = labeled_fraction * np.prod(seg.shape, dtype=np.int64)

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
