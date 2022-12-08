import numpy as np
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager


class SparseSegPixelWisePreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.annotated_pixels_percent = 0.03  # 3%

    def modify_seg_fn(self, seg: np.ndarray, plans: dict, dataset_json: dict, configuration: str) -> np.ndarray:
        seg = seg[0]
        label_manager = get_labelmanager(plans, dataset_json)
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label
        use_mask = np.random.random(
            seg.shape) < self.annotated_pixels_percent  # create a mask where 3% of the pixels are True
        seg_new[use_mask] = seg[use_mask]
        return seg_new[None]


class SparseSegPixelWisePreprocessor10(SparseSegPixelWisePreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.annotated_pixels_percent = 0.1


class SparseSegPixelWisePreprocessor30(SparseSegPixelWisePreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.annotated_pixels_percent = 0.3


class SparseSegPixelWisePreprocessor5(SparseSegPixelWisePreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.annotated_pixels_percent = 0.05


