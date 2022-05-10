from typing import Tuple, Union, List

from nnunetv2.utilities.utils import extract_unique_classes_from_dataset_json_labels


def handle_labels(dataset_json) -> Tuple[List, Union[List, None], Union[int, None]]:
    # first we need to check if we have to run region-based training
    region_needed = any([isinstance(i, tuple) and len(i) > 1 for i in dataset_json['labels'].values()])
    if region_needed:
        assert 'regions_class_order' in dataset_json.keys(), 'if region-based training is requested via ' \
                                                                  'dataset.json then you need to define ' \
                                                                  'regions_class_order as well, ' \
                                                                  'see documentation!'  # TODO add this
        regions = list(dataset_json['labels'].values())
        assert len(dataset_json['regions_class_order']) == len(regions), 'regions_class_order must have ans ' \
                                                                              'many entries as there are ' \
                                                                              'regions'
    else:
        regions = None
    all_labels = extract_unique_classes_from_dataset_json_labels(dataset_json['labels'])

    ignore_label = dataset_json['labels'].get('ignore')
    if ignore_label is not None:
        assert isinstance(ignore_label, int), f'Ignore label has to be an integer. It cannot be a region ' \
                                              f'(list/tuple). Got {type(ignore_label)}.'
    return all_labels, regions, ignore_label
