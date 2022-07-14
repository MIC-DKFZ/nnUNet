from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers


class nnUNetDataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_cases_properties_loading_threshold: int = 1000,
                 folder_with_segs_from_previous_stage: str = None):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_cases_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        TODO check all this
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """
        super().__init__()
        print('loading dataset')
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()
        # we ned to use super().__getitem__ so that we don't end up in a recursion problem because of our custom
        # implementation of __getitem__
        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, "%s.npz" % c)
            self.dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, "%s.npz" % c)

        if len(case_identifiers) <= num_cases_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        entry = self[key]
        if isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
        else:
            data = np.load(entry['data_file'])['data']
        if isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
        else:
            seg = np.load(entry['data_file'])['seg']
        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev))
        return data, seg, entry['properties']


if __name__ == '__main__':
    # this is a mini test. Todo: We can move this to tests in the future (requires simulated dataset)

    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset003_Liver/3d_lowres'
    ds = nnUNetDataset(folder, num_cases_properties_loading_threshold=0) # this should not load the properties!
    # this SHOULD HAVE the properties
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # amazing. I am the best.

    # this should have the properties
    ds = nnUNetDataset(folder, num_cases_properties_loading_threshold=1000)
    # now rename the properties file so that it doesnt exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now we should still be able to access the properties because they have already been loaded
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # move file back
    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

    # this should not have the properties
    ds = nnUNetDataset(folder, num_cases_properties_loading_threshold=0)
    # now rename the properties file so that it doesnt exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now this should crash
    try:
        ks = ds['liver_0'].keys()
        raise RuntimeError('we should not have come here')
    except FileNotFoundError:
        print('all good')
        # move file back
        shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

