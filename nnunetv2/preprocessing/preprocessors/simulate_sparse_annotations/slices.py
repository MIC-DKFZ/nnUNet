import numpy as np

from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager


class SparseSegSliceWiseOrth30Preprocessor(DefaultPreprocessor):
    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        seg = seg[0]
        every_nth_slice = 30
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label
        seg_new[:, :, ::every_nth_slice] = seg[:, :, ::every_nth_slice]
        seg_new[:, ::every_nth_slice] = seg[:, ::every_nth_slice]
        seg_new[::every_nth_slice] = seg[::every_nth_slice]
        return seg_new[None]


class SparseSegSliceWiseOrth10Preprocessor(DefaultPreprocessor):
    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        seg = seg[0]
        every_nth_slice = 10
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label
        seg_new[:, :, ::every_nth_slice] = seg[:, :, ::every_nth_slice]
        seg_new[:, ::every_nth_slice] = seg[:, ::every_nth_slice]
        seg_new[::every_nth_slice] = seg[::every_nth_slice]
        return seg_new[None]


class SparseSegSliceRandomOrth(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.percent_of_slices = 0.03

    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str, percent_of_slices: float = None) -> np.ndarray:
        if percent_of_slices is None:
            percent_of_slices = self.percent_of_slices
        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label
        x, y, z = seg.shape
        # x
        num_slices = max(1, round(x * percent_of_slices))
        selected_slices = np.random.choice(x, num_slices)
        seg_new[selected_slices] = seg[selected_slices]
        # y
        num_slices = max(1, round(y * percent_of_slices))
        selected_slices = np.random.choice(y, num_slices)
        seg_new[:, selected_slices] = seg[:, selected_slices]
        # z
        num_slices = max(1, round(z * percent_of_slices))
        selected_slices = np.random.choice(z, num_slices)
        seg_new[:, :, selected_slices] = seg[:, :, selected_slices]
        return seg_new[None]


class SparseSegSliceRandomOrth3(SparseSegSliceRandomOrth):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.percent_of_slices = 0.01


class SparseSegSliceRandomOrth5(SparseSegSliceRandomOrth):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.percent_of_slices = 0.05 / 3


class SparseSegSliceRandomSmartOrth(DefaultPreprocessor):
    """
    WE DO NOT USE THIS
    because always picking the set of orth slices that has the most pixel of a class may case some areas of the object never to be seen
    """
    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        # for each class, take one set of orthogonal slices where the most pixels of this class will be visible.
        # then take the rest of slices randomly (min 1 per axis)

        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label

        # this code finds the index where the maximum number of pixels of a class are present if three orthogonal
        # slices were taken at this point.
        # this is cool because the trivial approach (center of mass) would fail if objects are dispersed over the
        # entire image.
        # this algorithm is noice
        # https://external-preview.redd.it/qWUzXIlT_9lzmhBMexIzGT_8DtpsaqTT7Fo-_6a4W7o.gif?format=mp4&s=7550c9fc58fa8ef62b823e491992d0f2f7d7c84c
        workon = label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions
        num_orth_slices_taken = 0
        for w in workon:
            mask = region_or_label_to_mask(seg, w)
            if np.num(mask) == 0:
                continue
            acc = np.zeros_like(mask, dtype=np.int64)
            acc += mask.sum(0, keepdims=True)
            acc += mask.sum(1, keepdims=True)
            acc += mask.sum(2, keepdims=True)
            idx = np.unravel_index(acc.argmax(), acc.shape)
            seg_new[idx[0]] = seg[idx[0]]
            seg_new[:, idx[1]] = seg[:, idx[1]]
            seg_new[:, :, idx[2]] = seg[:, :, idx[2]]
            num_orth_slices_taken += 1

        percent_of_slices = 0.03
        x, y, z = seg.shape
        # x
        num_slices = min(1, round(x * percent_of_slices) - num_orth_slices_taken)
        selected_slices = np.random.choice(x, num_slices)
        seg_new[selected_slices] = seg[selected_slices]
        # y
        num_slices = min(1, round(y * percent_of_slices) - num_orth_slices_taken)
        selected_slices = np.random.choice(y, num_slices)
        seg_new[:, selected_slices] = seg[:, selected_slices]
        # z
        num_slices = min(1, round(z * percent_of_slices) - num_orth_slices_taken)
        selected_slices = np.random.choice(z, num_slices)
        seg_new[:, :, selected_slices] = seg[:, :, selected_slices]
        return seg_new[None]


class SparseSegSliceRandomSmart2Orth(DefaultPreprocessor):
    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        # for each class, take one set of orthogonal slices at a random location
        # then take the rest of slices randomly (min 1 per axis)

        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label

        locs = DefaultPreprocessor._sample_foreground_locations(seg,
                                                                label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions,
                                                                seed=None, verbose=False)
        # pick random pixel belonging to a class
        num_orth_slices_taken = 0
        for c in locs:
            if len(locs[c]) > 0:
                x, y, z = [int(i) for i in locs[c].astype(float)[np.random.choice(len(locs[c]))]]
                seg_new[x] = seg[x]
                seg_new[:, y] = seg[:, y]
                seg_new[:, :, z] = seg[:, :, z]
                num_orth_slices_taken += 1

        percent_of_slices = 0.03
        x, y, z = seg.shape
        # x
        num_slices = max(1, round(x * percent_of_slices) - num_orth_slices_taken)
        selected_slices = np.random.choice(x, num_slices)
        seg_new[selected_slices] = seg[selected_slices]
        # y
        num_slices = max(1, round(y * percent_of_slices) - num_orth_slices_taken)
        selected_slices = np.random.choice(y, num_slices)
        seg_new[:, selected_slices] = seg[:, selected_slices]
        # z
        num_slices = max(1, round(z * percent_of_slices) - num_orth_slices_taken)
        selected_slices = np.random.choice(z, num_slices)
        seg_new[:, :, selected_slices] = seg[:, :, selected_slices]
        return seg_new[None]


class RandomlyOrientedSlicesWithOversampling(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.target_percent_annotated_slices_per_axis = 0.03  # 10% annotation total

    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        # for each class, take one set of orthogonal slices at a random location
        # then take the rest of slices randomly (min 1 per axis)

        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label

        target_num_slices_per_axis = [int(round(self.target_percent_annotated_slices_per_axis * i)) for i in seg.shape]
        taken_num_slices = [0] * len(seg.shape)

        locs = DefaultPreprocessor._sample_foreground_locations(seg,
                                                                label_manager.foreground_labels if not label_manager.has_regions else label_manager.foreground_regions,
                                                                seed=None, verbose=False)
        for c in locs:
            if len(locs[c]) > 0:
                # on average one slice per class
                if np.random.uniform() < 0.33:
                    x, y, z = [int(i) for i in locs[c][np.random.choice(len(locs[c]))]]
                    seg_new[x] = seg[x]
                    taken_num_slices[0] += 1
                if np.random.uniform() < 0.33:
                    x, y, z = [int(i) for i in locs[c][np.random.choice(len(locs[c]))]]
                    seg_new[:, y] = seg[:, y]
                    taken_num_slices[1] += 1
                if np.random.uniform() < 0.33:
                    x, y, z = [int(i) for i in locs[c][np.random.choice(len(locs[c]))]]
                    seg_new[:, :, z] = seg[:, :, z]
                    taken_num_slices[2] += 1

        x, y, z = seg.shape
        # x
        num_slices = max(1, target_num_slices_per_axis[0] - taken_num_slices[0])
        selected_slices = np.random.choice(x, num_slices)
        seg_new[selected_slices] = seg[selected_slices]
        # y
        num_slices = max(1, target_num_slices_per_axis[1] - taken_num_slices[1])
        selected_slices = np.random.choice(y, num_slices)
        seg_new[:, selected_slices] = seg[:, selected_slices]
        # z
        num_slices = max(1, target_num_slices_per_axis[2] - taken_num_slices[2])
        selected_slices = np.random.choice(z, num_slices)
        seg_new[:, :, selected_slices] = seg[:, :, selected_slices]
        return seg_new[None]


class RandomlyOrientedSlicesWithOversampling5(RandomlyOrientedSlicesWithOversampling):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.target_percent_annotated_slices_per_axis = 0.05 / 3


class RandomlyOrientedSlicesWithOversampling3(RandomlyOrientedSlicesWithOversampling):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.target_percent_annotated_slices_per_axis = 0.01

