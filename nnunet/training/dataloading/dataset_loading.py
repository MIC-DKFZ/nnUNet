#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
from batchgenerators.augmentations.utils import random_crop_2D_image_batched, pad_nd_image
import numpy as np
from batchgenerators.dataloading import SlimDataLoaderBase
from multiprocessing import Pool
from nnunet.paths import preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *


def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


def get_case_identifiers_from_raw_folder(folder):
    case_identifiers = np.unique([i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz") and (i.find("segFromPrevStage") == -1)])
    return case_identifiers


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def save_as_npz(args):
    if not isinstance(args, tuple):
        key = "data"
        npy_file = args
    else:
        npy_file, key = args
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def unpack_dataset(folder, threads=8, key="data"):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key]*len(npz_files)))
    p.close()
    p.join()


def pack_dataset(folder, threads=8, key="data"):
    p = Pool(threads)
    npy_files = subfiles(folder, True, None, ".npy", True)
    p.map(save_as_npz, zip(npy_files, [key]*len(npy_files)))
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_case_identifiers(folder)
    npy_files = [join(folder, i+".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if isfile(i)]
    for n in npy_files:
        os.remove(n)


def load_dataset(folder):
    # we don't load the actual data but instead return the filename to the np file. the properties are loaded though
    case_identifiers = get_case_identifiers(folder)
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz"%c)
        with open(join(folder, "%s.pkl"%c), 'rb') as f:
            dataset[c]['properties'] = pickle.load(f)
        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz"%c)
    return dataset


def crop_2D_image_force_fg(img, crop_size, force_class=None):
    """
    img must be [c, x, y]
    img[-1] must be the segmentation with segmentation>0 being foreground
    :param img:
    :param crop_size:
    :return:
    """
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 1)
    else:
        assert len(crop_size) == (len(
            img.shape) - 1), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    # we need to find the center coords that we can crop to without exceeding the image border
    lb_x = crop_size[0] // 2
    ub_x = img.shape[1] - crop_size[0] // 2 - crop_size[0] % 2
    lb_y = crop_size[1] // 2
    ub_y = img.shape[2] - crop_size[1] // 2 - crop_size[1] % 2

    foreground_classes = np.unique(img[-1])
    foreground_classes = foreground_classes[foreground_classes > 0]
    if len(foreground_classes) == 0 or (0 in foreground_classes.shape):
        foreground_classes = [0]

    if force_class is None or force_class not in foreground_classes:
        chosen_class = np.random.choice(foreground_classes)
    else:
        chosen_class = force_class
    foreground_voxels = np.array(np.where(img[-1] == chosen_class))
    if np.any(np.array(foreground_voxels.shape) == 0):
        selected_center_voxel = (np.random.random_integers(lb_x, ub_x),
                                 np.random.random_integers(lb_y, ub_y))
    else:
        selected_center_voxel = foreground_voxels[:, np.random.choice(foreground_voxels.shape[1])]

    selected_center_voxel = np.array(selected_center_voxel)
    for i in range(2):
        selected_center_voxel[i] = max(crop_size[i]//2, selected_center_voxel[i])
        selected_center_voxel[i] = min(img.shape[i+1] - crop_size[i]//2 - crop_size[i] % 2, selected_center_voxel[i])

    result = img[:, (selected_center_voxel[0] - crop_size[0]//2):(selected_center_voxel[0] + crop_size[0]//2 + crop_size[0] % 2),
             (selected_center_voxel[1] - crop_size[1]//2):(selected_center_voxel[1] + crop_size[1]//2 + crop_size[1] % 2)]
    return result


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        """
        This is the basic data loader for 3D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With htis strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = []
        seg = []
        case_properties = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            case_properties.append(self._data[i]['properties'])
            # cases are stores as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            # If we are doing the cascade then we will also need to load the segmentation of the previous stage and
            # concatenate it. Here it will be concatenates to the segmentation because the augmentations need to be
            # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
            # the last channel of the data
            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                # we theoretically support several possible previsous segmentations from which only one is sampled. But
                # in practice this feature was never used so it's always only one segmentation
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key+1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            # do you trust me? You better do. Otherwise you'll have to go through this mess and honestly there are
            # better things you could do right now

            # (above) documentation of the day. Nice. Even myself coming back 1 months later I have not friggin idea
            # what's going on. I keep the above documentation just for fun but attempt to make things clearer now

            need_to_pad = self.need_to_pad
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d+1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d+1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint
            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.array(self._data[i]['properties']['classes'])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = 0
                else:
                    selected_class = np.random.choice(foreground_classes)

                # we could precompute that to save CPU time but I am too lazy right now
                voxels_of_that_class = np.argwhere(case_all_data[-1] == selected_class)

                # voxels_of_that_class should always contain some voxels we chose selected_class from the classes that
                # are actually present in the case. There is however one slight chance that foreground_classes includes
                # a class that is not anymore present in the preprocessed data. That is because np.unique is called on
                # the data in their original resolution but the data/segmentations here are present in some resampled
                # resolution. In extremely rare cases (and if the ground truth was bad = single annoteted pixels) then
                # a class can get lost in the preprocessing. This is not a problem at all because that segmentation was
                # most likely faulty anyways but it may lead to crashes here.
                if len(voxels_of_that_class) != 0:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the selected class is indeed not present then we fall back to random cropping. We can do that
                    # because this case is extremely rare.
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                             valid_bbox_y_lb:valid_bbox_y_ub,
                                             valid_bbox_z_lb:valid_bbox_z_ub]
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                                                     valid_bbox_y_lb:valid_bbox_y_ub,
                                                                     valid_bbox_z_lb:valid_bbox_z_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0,0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0,0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                         'constant', **{'constant_values':-1})
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = np.pad(seg_from_previous_stage, ((0,0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                                 'constant', **{'constant_values': 0})
                case_all_data_segonly = np.concatenate((case_all_data_segonly, seg_from_previous_stage), 0)

            data.append(case_all_data_donly[None])
            seg.append(case_all_data_segonly[None])

        # this is bad. Slow. Better preallocate data and set with np.zeros(). But this way we don't have to know how
        # many color channels data has and how many seg channels there are
        data = np.vstack(data)
        seg = np.vstack(seg)

        return {'data':data, 'seg':seg, 'properties':case_properties, 'keys': selected_keys}


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, transpose=None,
                 oversample_foreground_percent=0.0, memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With htis strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        if transpose is not None:
            assert isinstance(transpose, (list, tuple)), "Transpose must be either None or be a tuple/list representing the new axis order (3 ints)"
        self.transpose = transpose
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides

    @property
    def all_possible_slices(self):
        try:
            return self._all_possible_slices
        except AttributeError:
            slices = []
            for key in sorted(self.list_of_keys):
                shape = self._data[key]["properties"]["size_after_resampling"]
                for s in range(shape[0]):
                    slices.append((key, s))
            self._all_possible_slices = slices
            return slices

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = []
        seg = []
        case_properties = []
        for j, i in enumerate(selected_keys):
            properties = self._data[i]['properties']
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            if self.transpose is not None:
                leading_axis = self.transpose[0]
            else:
                leading_axis = 0

            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[leading_axis + 1])
            else:
                # select one class, then select a slice that contains that class
                classes_in_slice_per_axis = properties.get("classes_in_slice_per_axis")
                possible_classes = np.unique(properties['classes'])
                possible_classes = possible_classes[possible_classes > 0]
                if len(possible_classes) > 0 and not (0 in possible_classes.shape):
                    selected_class = np.random.choice(possible_classes)
                else:
                    selected_class = 0
                if classes_in_slice_per_axis is not None:
                    valid_slices = classes_in_slice_per_axis[leading_axis][selected_class]
                else:
                    valid_slices = np.where(np.sum(case_all_data[-1] == selected_class, axis=[i for i in range(3) if i != leading_axis]))[0]
                if len(valid_slices) != 0:
                    random_slice = np.random.choice(valid_slices)
                else:
                    random_slice = np.random.choice(case_all_data.shape[leading_axis + 1])

            if self.pseudo_3d_slices == 1:
                if leading_axis == 0:
                    case_all_data = case_all_data[:, random_slice]
                elif leading_axis == 1:
                    case_all_data = case_all_data[:, :, random_slice]
                else:
                    case_all_data = case_all_data[:, :, :, random_slice]
                if self.transpose is not None and self.transpose[1] > self.transpose[2]:
                    case_all_data = case_all_data.transpose(0, 2, 1)
            else:
                assert leading_axis == 0, "pseudo_3d_slices works only without transpose for now!"
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            num_seg = 1

            # why we need this is a little complicated. It has to do with downstream random cropping during data
            # augmentation. Basically we will rotate the patch and then to a center crop of size self.final_patch_size.
            # Depending on the rotation, scaling and elastic deformation parameters, self.patch_size has to be large
            # enough to prevent border artifacts from being present in the final patch
            new_shp = None
            if np.any(self.need_to_pad) > 0:
                new_shp = np.array(case_all_data.shape[1:] + self.need_to_pad)
                if np.any(new_shp - np.array(self.patch_size) < 0):
                    new_shp = np.max(np.vstack((new_shp[None], np.array(self.patch_size)[None])), 0)
            else:
                if np.any(np.array(case_all_data.shape[1:]) - np.array(self.patch_size) < 0):
                    new_shp = np.max(
                        np.vstack((np.array(case_all_data.shape[1:])[None], np.array(self.patch_size)[None])), 0)
            if new_shp is not None:
                case_all_data_donly = pad_nd_image(case_all_data[:-num_seg], new_shp, self.pad_mode, kwargs=self.pad_kwargs_data)
                case_all_data_segnonly = pad_nd_image(case_all_data[-num_seg:], new_shp, 'constant', kwargs={'constant_values':-1})
                case_all_data = np.vstack((case_all_data_donly, case_all_data_segnonly))[None]
            else:
                case_all_data = case_all_data[None]

            if not force_fg:
                case_all_data = random_crop_2D_image_batched(case_all_data, tuple(self.patch_size))
            else:
                case_all_data = crop_2D_image_force_fg(case_all_data[0], tuple(self.patch_size), selected_class)[None]
            data.append(case_all_data[:, :-num_seg])
            seg.append(case_all_data[:, -num_seg:])
        data = np.vstack(data)
        seg = np.vstack(seg)
        keys = selected_keys
        return {'data':data, 'seg':seg, 'properties':case_properties, "keys": keys}



if __name__ == "__main__":
    t = "Task02_Heart"
    p = join(preprocessing_output_dir, t, "stage1")
    dataset = load_dataset(p)
    with open(join(join(preprocessing_output_dir, t), "plans_stage1.pkl"), 'rb') as f:
        plans = pickle.load(f)
    unpack_dataset(p)
    dl = DataLoader3D(dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33)
    dl = DataLoader3D(dataset, np.array(plans['patch_size']).astype(int), np.array(plans['patch_size']).astype(int), 2, oversample_foreground_percent=0.33)
    dl2d = DataLoader2D(dataset, (64, 64), np.array(plans['patch_size']).astype(int)[1:], 12, oversample_foreground_percent=0.33)
