from typing import List, Tuple, Union

from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


class ConvertSegmentationToRegionsTransform(AbstractTransform):
    def __init__(self, regions: Union[List, Tuple],
                 seg_key: str = "seg", output_key: str = "seg", seg_channel: int = 0):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region,
        example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is not None:
            b, c, *shape = seg.shape
            region_output = np.zeros((b, len(self.regions), *shape), dtype=bool)
            for region_id, region_labels in enumerate(self.regions):
                region_output[:, region_id] |= np.isin(seg[:, self.seg_channel], region_labels)
            data_dict[self.output_key] = region_output.astype(np.uint8, copy=False)
        return data_dict

