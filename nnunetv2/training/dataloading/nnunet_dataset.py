import os
from typing import List, Union, Tuple

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers

from torch.utils.data import Dataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager

class nnUNetDataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_images_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """
        super().__init__()
        # print('loading dataset')
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, f"{c}.npz")
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        # print(f'nnUNetDataset.keep_files_open: {self.keep_files_open}')

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
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
                # print('saving open data file')
        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
                # print('saving open seg file')
        else:
            seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        return data, seg, entry['properties']

class nnUNetPytorchDataset(Dataset, nnUNetDataset):
    '''
    Pytorch version of the nnUNet Dataset.
    Goal is to implement the same functionality, but be able to use Pytorch's in-built
    parallel dataloading abilities without having to use the Multi-Thread Augmenter from
    batchgenerators.
    For now, the only dependency we want on Batch Generators is 
    (1) Transforms 

    Next step -> I would want to also do away with dependency on Transforms

    For now -> Using this as an inherited class -> Because nnUNet class already has a lot of useful 
    logic for how data is being loaded. 
    Also this is still in the nnunet library so it is ok. 
    '''     
    def __init__(self, folder: str, patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,                 
                 case_identifiers: List[str] = None,                 
                 oversample_foreground_percent: float = 0.0,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        # Initialize 
        nnUNetDataset.__init__(self, folder, case_identifiers, num_images_properties_loading_threshold,
                 folder_with_segs_from_previous_stage)
        Dataset.__init__(self)

        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        self.has_ignore = label_manager.has_ignore_label
        self.annotated_classes_key = tuple(label_manager.all_labels)
        self.oversample_foreground_percent = oversample_foreground_percent
    
    def _get_case_dict(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret
    
    def load_case(self, idx):
        # This is different because Pytorch indexes with numbers not keys
        key = self.keys()[idx]
        entry = self._get_case_dict[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
                # print('saving open data file')
        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
                # print('saving open seg file')
        else:
            seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        return data, seg, entry['properties']

    # Refer to nnUNetDataLoader3D in `dataloading/data_loader_3d.py` to see why I wrote it this way
    def __getitem__(self, idx):
        '''
        Based on:
        (1) if we are foreground sampling
        (2) forgeground sampling percentage
        (3) initial patch size - I AM NOT YET SURE WHAT EXACTLY THIS IS!
        (4) final patch size (which comes from the plans file)

        we crop and get an appropriate patch of the right size 

        I am also not sure at what point and how exactly nnUNet applies transform 
        What I do know is that the taransforms are applied at the batch levele in the producer function. 

        Look into that -> That will give insight on how to apply transforms .. here or via the dataloader???         

        Variable to give as input (delete this after):
        - oversample_foreground_percent
        - patch_size
        - has_ignore

        Functions from the nnUNet Dataloadert that need to be moved to this dataset class. 
        - get_bbox
        - need to pad (which is) - self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)

        '''                
        
        # Read in ENTIRE CT and Segmentation from Disk and the properties 
        data, seg, properties = self.load_case(idx)

        shape = data.shape[1:]
        dim = len(shape)

        # Randomly choose oversample - This is different from what nnUNet defaults to
        if np.random.uniform() < self.oversample_foreground_percent:
            force_fg = True
        else:
            force_fg = False

        # This step gets the bounding box for the patch - and uses the force_fg to determine if we need to oversample
        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

        valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

        this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        data = data[this_slice]

        this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        seg = seg[this_slice]        

        padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]

        data_padded = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
        seg_padded = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        # Apply transforms here !! - Not done yet :(

        return data_padded, seg_padded


    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs




if __name__ == '__main__':
    # this is a mini test. Todo: We can move this to tests in the future (requires simulated dataset)

    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset003_Liver/3d_lowres'
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0) # this should not load the properties!
    # this SHOULD HAVE the properties
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # amazing. I am the best.

    # this should have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=1000)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now we should still be able to access the properties because they have already been loaded
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # move file back
    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

    # this should not have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now this should crash
    try:
        ks = ds['liver_0'].keys()
        raise RuntimeError('we should not have come here')
    except FileNotFoundError:
        print('all good')
        # move file back
        shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

