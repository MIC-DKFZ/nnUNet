import os
import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from threadpoolctl import threadpool_limits

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class nnUNetDataLoader(DataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        """
        If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        returning the batch
        """
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        # this is used by DataLoader for sampling train cases!
        self.indices = data.identifiers

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms
        self._pre_data_t = None
        self._pre_seg_t = None

        self._post_images_buf = None
        self._post_images_sig = None  # (dtype, shape_tuple)

        self._post_segs_buf = None
        self._post_segs_sig = None  # either ("tensor", dtype, shape) or ("list", ((dtype, shape), ...))

    def _get_pre_buffers(self):
        if self._pre_data_t is None or tuple(self._pre_data_t.shape) != tuple(self.data_shape):
            self._pre_data_t = torch.empty(self.data_shape, dtype=torch.float32)  # CPU
        if self._pre_seg_t is None or tuple(self._pre_seg_t.shape) != tuple(self.seg_shape):
            self._pre_seg_t = torch.empty(self.seg_shape, dtype=torch.int16)  # CPU
        return self._pre_data_t, self._pre_seg_t

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, seg_prev, properties = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        return data_shape, seg_shape

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
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
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

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
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

    def generate_train_batch(self):
        try:
            selected_keys = self.get_indices()

            # Preallocate and reuse per worker pre-transform buffers (CPU)
            data_all, seg_all = self._get_pre_buffers()

            for j, i in enumerate(selected_keys):
                force_fg = self.get_do_oversample(j)

                # data/seg/seg_prev can be numpy arrays or blosc2 arrays/memmaps
                data, seg, seg_prev, properties = self._data.load_case(i)

                shape = data.shape[1:]
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
                bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

                # Crop+pad returns NumPy. Copy into preallocated torch buffers explicitly.
                data_np = crop_and_pad_nd(data, bbox, 0)
                data_all[j].copy_(torch.as_tensor(data_np, dtype=torch.float32))

                seg_np = crop_and_pad_nd(seg, bbox, -1)
                seg_t = torch.as_tensor(seg_np, dtype=torch.int16)  # (Cseg, *spatial)
                seg_all[j, :seg_t.shape[0]].copy_(seg_t)

                if seg_prev is not None:
                    prev_np = crop_and_pad_nd(seg_prev, bbox, -1)
                    prev_t = torch.as_tensor(prev_np, dtype=torch.int16)

                    # Normalize seg_prev to (*spatial) and write into the reserved extra channel.
                    if prev_t.ndim == seg_t.ndim:
                        if prev_t.shape[0] == 1:
                            prev_t = prev_t[0]
                        else:
                            raise RuntimeError(
                                f"seg_prev has {prev_t.shape[0]} channels, but seg_shape reserves only one extra channel"
                            )
                    elif prev_t.ndim != seg_t.ndim - 1:
                        raise RuntimeError(
                            f"Unexpected seg_prev ndim {prev_t.ndim}. Expected {seg_t.ndim - 1} (no channel) "
                            f"or {seg_t.ndim} (singleton channel)."
                        )

                    seg_all[j, seg_t.shape[0]].copy_(prev_t)

            if self.patch_size_was_2d:
                data_all = data_all[:, :, 0]
                seg_all = seg_all[:, :, 0]

            if self.transforms is None:
                return {"data": data_all, "target": seg_all, "keys": selected_keys}

            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    pre_data, pre_seg = data_all, seg_all  # CPU tensors

                    # Probe first sample to determine output structure.
                    tmp0 = self.transforms(image=pre_data[0], segmentation=pre_seg[0])
                    img0 = tmp0["image"]
                    seg0 = tmp0["segmentation"]

                    # Lazily allocate or reuse persistent post-transform buffers.
                    # Images
                    img_sig = (img0.dtype, tuple(img0.shape))
                    if getattr(self, "_post_images_buf", None) is None or getattr(self, "_post_images_sig",
                                                                                  None) != img_sig:
                        self._post_images_buf = torch.empty(
                            (self.batch_size, *img0.shape), dtype=img0.dtype, device="cpu"
                        )
                        self._post_images_sig = img_sig

                    out_images = self._post_images_buf
                    out_images[0].copy_(img0)

                    # Segmentation: tensor or list of tensors
                    if isinstance(seg0, list):
                        seg_sig = ("list", tuple((s.dtype, tuple(s.shape)) for s in seg0))
                        if getattr(self, "_post_segs_buf", None) is None or getattr(self, "_post_segs_sig",
                                                                                    None) != seg_sig:
                            self._post_segs_buf = [
                                torch.empty((self.batch_size, *s.shape), dtype=s.dtype, device="cpu") for s in seg0
                            ]
                            self._post_segs_sig = seg_sig

                        out_segs = self._post_segs_buf
                        for k, s0 in enumerate(seg0):
                            out_segs[k][0].copy_(s0)
                    else:
                        seg_sig = ("tensor", seg0.dtype, tuple(seg0.shape))
                        if getattr(self, "_post_segs_buf", None) is None or getattr(self, "_post_segs_sig",
                                                                                    None) != seg_sig:
                            self._post_segs_buf = torch.empty(
                                (self.batch_size, *seg0.shape), dtype=seg0.dtype, device="cpu"
                            )
                            self._post_segs_sig = seg_sig

                        out_segs = self._post_segs_buf
                        out_segs[0].copy_(seg0)

                    # Fill remaining samples into persistent buffers.
                    for b in range(1, self.batch_size):
                        tmp = self.transforms(image=pre_data[b], segmentation=pre_seg[b])

                        out_images[b].copy_(tmp["image"])

                        segb = tmp["segmentation"]
                        if isinstance(out_segs, list):
                            if not isinstance(segb, list) or len(segb) != len(out_segs):
                                raise RuntimeError("Transform returned inconsistent segmentation structure across batch")
                            for k in range(len(out_segs)):
                                out_segs[k][b].copy_(segb[k])
                        else:
                            if isinstance(segb, list):
                                raise RuntimeError(
                                    "Transform returned list segmentation for later sample but tensor for first sample")
                            out_segs[b].copy_(segb)

                    return {"data": out_images, "target": out_segs, "keys": selected_keys}
        except Exception as e:
            print('EXCEPTION:', e)
            raise e


if __name__ == '__main__':
    folder = join(nnUNet_preprocessed, 'Dataset002_Heart', 'nnUNetPlans_3d_fullres')
    ds = nnUNetDatasetBlosc2(folder)  # this should not load the properties!
    pm = PlansManager(join(folder, os.pardir, 'nnUNetPlans.json'))
    lm = pm.get_label_manager(load_json(join(folder, os.pardir, 'dataset.json')))
    dl = nnUNetDataLoader(ds, 5, (16, 16, 16), (16, 16, 16), lm,
                          0.33, None, None)
    a = next(dl)
