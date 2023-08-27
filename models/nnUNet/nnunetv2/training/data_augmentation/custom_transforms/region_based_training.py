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
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for region_id, region_source_labels in enumerate(self.regions):
                    if not isinstance(region_source_labels, (list, tuple)):
                        region_source_labels = (region_source_labels, )
                    for label_value in region_source_labels:
                        region_output[b, region_id][seg[b, self.seg_channel] == label_value] = 1
            data_dict[self.output_key] = region_output
        return data_dict
