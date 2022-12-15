from typing import Union, Tuple

import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5, nnUNetTrainerDA5ord0


class nnUNetDataLoaderBaseBetterIgnSampling(nnUNetDataLoaderBase):
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
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in
                       eligible_classes_or_regions]
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
                        (overwrite_class is None or (
                                    overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                #################################################################
                if self.has_ignore and not force_fg:
                    # # random offset for selected voxel
                    # orig = deepcopy(selected_voxel)
                    allowed_max_neg_offset = [min(s, p // 2) for s, p in zip(selected_voxel[1:], self.patch_size)]
                    allowed_max_pos_offset = [min(d - s, p // 2) for s, p, d in
                                              zip(selected_voxel[1:], self.patch_size, data_shape)]
                    for d in range(len(self.patch_size)):
                         selected_voxel[d + 1] += np.random.randint(-allowed_max_neg_offset[d], allowed_max_pos_offset[d])
                    # offset = deepcopy(selected_voxel)
                    # # make sure selected voxels are within image boundaries
                    # selected_voxel = [selected_voxel[0]] + [max(0, i) for i in selected_voxel[1:]]
                    # selected_voxel = [selected_voxel[0]] + [min(d, i) for d, i in zip(data_shape, selected_voxel[1:])]
                    # # corr = deepcopy(selected_voxel)
                    # print(f'orig {orig}, offset {offset}, corr {corr}, data shape {data_shape}')
                #################################################################

                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs


# the following class is evil!
class nnUNetDataLoader2DBetterIgnSampling(nnUNetDataLoader2D):
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        return nnUNetDataLoaderBaseBetterIgnSampling.get_bbox(self, data_shape, force_fg, class_locations,
                                                              overwrite_class, verbose)


# the following class is evil!
class nnUNetDataLoader3DBetterIgnSampling(nnUNetDataLoader3D):
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        return nnUNetDataLoaderBaseBetterIgnSampling.get_bbox(self, data_shape, force_fg, class_locations,
                                                              overwrite_class, verbose)


class nnUNetTrainer_betterIgnoreSampling(nnUNetTrainer):
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DBetterIgnSampling(dataset_tr,
                                                        self.batch_size,
                                                        initial_patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2DBetterIgnSampling(dataset_val,
                                                         self.batch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.label_manager,
                                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                                         sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3DBetterIgnSampling(dataset_tr,
                                                        self.batch_size,
                                                        initial_patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3DBetterIgnSampling(dataset_val,
                                                         self.batch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.label_manager,
                                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                                         sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val


class nnUNetTrainerDA5_betterIgnoreSampling(nnUNetTrainerDA5):
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DBetterIgnSampling(dataset_tr,
                                                        self.batch_size,
                                                        initial_patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2DBetterIgnSampling(dataset_val,
                                                         self.batch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.label_manager,
                                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                                         sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3DBetterIgnSampling(dataset_tr,
                                                        self.batch_size,
                                                        initial_patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3DBetterIgnSampling(dataset_val,
                                                         self.batch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.label_manager,
                                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                                         sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val


class nnUNetTrainerDA5ord0_betterIgnoreSampling(nnUNetTrainerDA5ord0):
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DBetterIgnSampling(dataset_tr,
                                                        self.batch_size,
                                                        initial_patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2DBetterIgnSampling(dataset_val,
                                                         self.batch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.label_manager,
                                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                                         sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3DBetterIgnSampling(dataset_tr,
                                                        self.batch_size,
                                                        initial_patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3DBetterIgnSampling(dataset_val,
                                                         self.batch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.label_manager,
                                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                                         sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val


class nnUNetTrainer_betterIgnoreSampling_10epochs(nnUNetTrainer_betterIgnoreSampling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda'):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10
