from typing import List

from batchgenerators.transforms.abstract_transforms import AbstractTransform


class MaskTransform(AbstractTransform):
    def __init__(self, apply_to_channels: List[int], mask_idx_in_seg: int = 0, set_outside_to: int = 0,
                 data_key: str = "data", seg_key: str = "seg"):
        """
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!
        """
        self.apply_to_channels = apply_to_channels
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        mask = data_dict[self.seg_key][:, self.mask_idx_in_seg] < 0
        for c in self.apply_to_channels:
            data_dict[self.data_key][:, c][mask] = self.set_outside_to
        return data_dict
