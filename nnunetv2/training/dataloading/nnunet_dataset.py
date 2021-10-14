import shutil
import shutil
import sys

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle
from nnunetv2.training.dataloading.utils import get_case_identifiers


class nnUNetDataset(dict):
    def __init__(self, folder: str, num_cases_properties_loading_threshold: int = 1000):
        """
        Make no mistake. This is not a torchvision dataset. It's just a dict, bro

        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are again dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - data_file -> the full path to the npz file associated with the training case
        - properties_file -> the pkl file containing the case properties
        In addition, if the total number of cases is < num_cases_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can be too large causing too much RAM utilization. In that
        case is it better to load on the fly.
        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - properties -> dict
        """
        super().__init__()
        print('loading dataset')
        case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()
        # we ned to use super().__getitem__ so that we don't end up in a recursion problem because of our custom
        # implementation of __getitem__
        for c in case_identifiers:
            self[c] = {}
            super().__getitem__(c)['data_file'] = join(folder, "%s.npz" % c)
            super().__getitem__(c)['properties_file'] = join(folder, "%s.pkl" % c)

        if len(case_identifiers) <= num_cases_properties_loading_threshold:
            print('loading all case properties')
            for i in self.keys():
                super().__getitem__(i)['properties'] = load_pickle(super().__getitem__(i)['properties_file'])

    def __getitem__(self, item):
        ret = {**super().__getitem__(item)}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret


if __name__ == '__main__':
    # this is a mini test. Todo: We can move this to tests in the future (requires simulated dataset)

    folder = '/media/fabian/data/nnUNet_preprocessed/Task003_Liver/3d_lowres'
    ds = nnUNetDataset(folder, 0) # this should not load the properties!
    # this should NOT have the properties
    ks = super(nnUNetDataset, ds).__getitem__('liver_0').keys()
    assert 'properties' not in ks
    # this SHOULD HAVE the properties
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # amazing. I am the best.

    # this should have the properties
    ds = nnUNetDataset(folder, 1000)
    # now rename the properties file so that it doesnt exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now we should still be able to access the properties because they have already been loaded
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # move file back
    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

    # this should not have the properties
    ds = nnUNetDataset(folder, 0)
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

