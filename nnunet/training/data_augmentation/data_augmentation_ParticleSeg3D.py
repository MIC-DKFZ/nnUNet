#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor, OneOfTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform

from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform
import numpy as np
import pickle
import random
import zarr
from nnunet.training.data_augmentation.slicer import slicer
from os.path import join
from skimage.measure import regionprops
import SimpleITK as sitk
import uuid
import time
from skimage.filters import gaussian

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


def get_ParticleSeg3D_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params,
                                   border_val_seg=-1,
                                   seeds_train=None, seeds_val=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                                   soft_ds=False,
                                   classes=None, pin_memory=True, regions=None,
                                   use_nondetMultiThreadedAugmenter: bool = False, raw_data_dir=None, preprocessed_data_dir=None):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    tr_transforms.append(RandParticleTouch(raw_data_dir, preprocessed_data_dir, "labelsTr_instance_zarr", "labelsTr_zarr", "regionprops.pkl"))

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    tr_transforms.append(BrightnessTransform(0,
                                             0.5,
                                             per_channel=True,
                                             p_per_sample=0.1,
                                             p_per_channel=0.5
                                             )
                         )

    tr_transforms.append(OneOfTransform(
        [
            ContrastAugmentationTransform(
                contrast_range=(0.5, 2),
                preserve_range=True,
                per_channel=True,
                data_key='data',
                p_per_sample=0.2,
                p_per_channel=0.5
            ),
            ContrastAugmentationTransform(
                contrast_range=(0.5, 2),
                preserve_range=False,
                per_channel=True,
                data_key='data',
                p_per_sample=0.2,
                p_per_channel=0.5
            ),
        ]
    ))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(
        GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(
        GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))

    tr_transforms.append(
        BrightnessGradientAdditiveTransform(
            lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
            (-0.5, 1.5),
            max_strength=lambda x, y: np.random.uniform(-5, -1) if np.random.uniform() < 0.5 else np.random.uniform(1, 5),
            mean_centered=False,
            same_for_all_channels=False,
            p_per_sample=0.3,
            p_per_channel=0.5
        )
    )

    tr_transforms.append(
        LocalGammaTransform(
            lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
            (-0.5, 1.5),
            lambda: np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4),
            same_for_all_channels=False,
            p_per_sample=0.3,
            p_per_channel=0.5
        )
    )

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("cascade_do_cascade_augmentations") is not None and params.get(
                "cascade_do_cascade_augmentations"):
            if params.get("cascade_random_binary_transform_p") > 0:
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size"),
                    p_per_label=params.get("cascade_random_binary_transform_p_per_label")))
            if params.get("cascade_remove_conn_comp_p") > 0:
                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                        key="data",
                        p_per_sample=params.get("cascade_remove_conn_comp_p"),
                        fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                        dont_do_if_covers_more_than_X_percent=params.get(
                            "cascade_remove_conn_comp_fill_with_other_class_p")))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_train = NonDetMultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                            params.get("num_cached_per_thread"), seeds=seeds_train,
                                                            pin_memory=pin_memory)
    else:
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                      params.get("num_cached_per_thread"),
                                                      seeds=seeds_train, pin_memory=pin_memory)
    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    # import IPython;IPython.embed()

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_val = NonDetMultiThreadedAugmenter(dataloader_val, val_transforms,
                                                          max(params.get('num_threads') // 2, 1),
                                                          params.get("num_cached_per_thread"),
                                                          seeds=seeds_val, pin_memory=pin_memory)
    else:
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                    max(params.get('num_threads') // 2, 1),
                                                    params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)

    return batchgenerator_train, batchgenerator_val


class RandParticleTouch(AbstractTransform):
    def __init__(self, raw_data_dir, preprocessed_data_dir, instance_seg_folder, border_core_seg_folder, metadata, repetitions=1):
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.instance_seg_folder = instance_seg_folder
        self.border_core_seg_folder = border_core_seg_folder
        self.repetitions = repetitions
        print("raw_data_dir: ", raw_data_dir)
        print("preprocessed_data_dir: ", preprocessed_data_dir)

        with open(join(self.raw_data_dir, metadata), 'rb') as handle:
            self.metadata = pickle.load(handle)

        min_size = 500
        filtered_metadata = {}
        for name in self.metadata.keys():
            filtered_metadata[name] = {}
            for label in self.metadata[name].keys():
                bbox = self.metadata[name][label]
                bbox = [[bbox[i], bbox[i + len(bbox) // 2]] for i in range(len(bbox) // 2)]
                bbox = np.asarray(bbox)
                size = np.prod(bbox[:, 1] - bbox[:, 0])
                if size >= min_size:
                    filtered_metadata[name][label] = bbox
        self.metadata = filtered_metadata

    def __call__(self, **data_dict):
        # t = time.localtime()
        # current_time = time.strftime("%H:%M:%S", t)
        # with open('/dkfz/cluster/gpu/checkpoints/OE0441/k539i/nnUNet/3d_fullres/Task310_particle_seg/nnUNetTrainerV2_touch_V1__nnUNetPlansv2.1/fold_0/ParticleTouchAug.txt', 'a') as f:
        #     f.write(current_time)
        for i in range(len(data_dict['data'])):
            img = data_dict['data'][i][0]
            seg = data_dict['seg'][i][0]
            instance_seg = zarr.open(join(self.raw_data_dir, self.instance_seg_folder, data_dict['keys'][i] + ".zarr"), mode='r')["arr_0"].astype(np.uint16)
            with open(join(self.preprocessed_data_dir, "nnUNetData_plans_v2.1_stage0", data_dict['keys'][i] + ".pkl"), 'rb') as handle:
                properties = pickle.load(handle)
            crop_offset = np.asarray(properties["crop_bbox"])[:, 0]
            instance_seg, negative_image = self.crop_pad_seg(instance_seg, crop_offset, data_dict['valid_bbox'][i], data_dict['bbox'][i],
                                                             data_dict['before_crop_shape'][i], data_dict['after_crop_shape'][i], data_dict['after_pad_shape'][i], data_dict['padding_tmp'][i], data_dict['shapy_shape'][i])
            # print("img: {}, border_core: {}, instance: {}, negative: {}".format(img.shape, seg.shape, instance_seg.shape, negative_image.shape))
            # try:
            #     assert np.array_equal(img.shape, seg.shape)
            #     assert np.array_equal(img.shape, instance_seg.shape)
            #     assert np.array_equal(img.shape, negative_image.shape)
            # except Exception as e:
            #     print("img: {}, border_core: {}, instance: {}, negative: {}, valid_bbox: {}, bbox: {}".format(img.shape, seg.shape, instance_seg.shape, negative_image.shape, data_dict['valid_bbox'][i], data_dict['bbox'][i]))
            #     raise e

            img, seg = self.do_particle_touch(img, seg, instance_seg, negative_image, data_dict['keys'][i])
            img = img[np.newaxis, ...]
            seg = seg[np.newaxis, ...]
            data_dict['data'][i], data_dict['seg'][i] = img, seg
        return data_dict

    def crop_pad_seg(self, instance_seg, crop_offset0, valid_bbox, bbox, before_crop_shape, after_crop_shape, after_pad_shape, padding_fabian, shapy_shape):
        # bbox_x_lb, bbox_x_ub, bbox_y_lb, bbox_y_ub, bbox_z_lb, bbox_z_ub = bbox
        # shape = instance_seg.shape
        shape = shapy_shape
        # instance_seg_shape_before_crop = instance_seg.shape
        # if not np.array_equal(instance_seg.shape, before_crop_shape[1:]):
        #     print("")
        valid_bbox = np.reshape(valid_bbox, (-1, 2))
        crop_offset = np.stack((crop_offset0, crop_offset0), axis=1)
        valid_bbox += crop_offset
        bbox = np.reshape(bbox, (-1, 2))
        # bbox_diff1 = bbox[:, 1] - bbox[:, 0]
        # bbox += crop_offset
        # bbox_diff2 = bbox[:, 1] - bbox[:, 0]
        # if not np.array_equal(bbox_diff2, np.asarray((205, 205, 205))):
        #     print("")
        instance_seg = instance_seg[slicer(instance_seg, valid_bbox)]
        instance_seg_shape_after_crop = instance_seg.shape
        # if not np.array_equal(instance_seg.shape, after_crop_shape[1:]):
        #     print("")
        # shape = instance_seg.shape
        # shape_tmp1 = instance_seg.shape
        negative_image = np.zeros(instance_seg.shape, dtype=np.uint8)
        # padding_tmp = ((-min(0, bbox[0][0]), max(bbox[0][1] - shape[0], 0)),
        #                             (-min(0, bbox[1][0]), max(bbox[1][1] - shape[1], 0)),
        #                             (-min(0, bbox[2][0]), max(bbox[2][1] - shape[2], 0)))
        # padding_tmp = ((-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
        #              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
        #              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0)))
        instance_seg = np.pad(instance_seg, (
                                    (-min(0, bbox[0][0]), max(bbox[0][1] - shape[0], 0)),
                                    (-min(0, bbox[1][0]), max(bbox[1][1] - shape[1], 0)),
                                    (-min(0, bbox[2][0]), max(bbox[2][1] - shape[2], 0))),
               "edge")
        negative_image = np.pad(negative_image, (
                                    (-min(0, bbox[0][0]), max(bbox[0][1] - shape[0], 0)),
                                    (-min(0, bbox[1][0]), max(bbox[1][1] - shape[1], 0)),
                                    (-min(0, bbox[2][0]), max(bbox[2][1] - shape[2], 0))),
               mode='constant', constant_values=1)
        # instance_seg_shape_after_pad = instance_seg.shape
        # if not np.array_equal(instance_seg.shape, after_pad_shape[1:]):
        #     print("")
        # shape_tmp2 = instance_seg.shape
        # if shape_tmp2 != (205, 205, 205):
        #     print("1: {}, 2: {}, 3: {}, padding: {}".format(shape, shape_tmp1, shape_tmp2, padding_tmp))
        return instance_seg, negative_image

    def do_particle_touch(self, patch_img, patch_seg, instance_seg, patch_negative, name):
        if not np.any(instance_seg):
            return patch_img, patch_seg
        # save_dir = "/home/k539i/Documents/datasets/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Task310_particle_seg/tmp"
        # id = str(uuid.uuid4())[:8]
        # sitk.WriteImage(sitk.GetImageFromArray(patch_img), join(save_dir, "{}_image_before.nii.gz".format(id)))
        # sitk.WriteImage(sitk.GetImageFromArray(patch_seg), join(save_dir, "{}_seg_before.nii.gz".format(id)))
        # patch_img_before = copy.deepcopy(patch_img)
        # patch_seg_before = copy.deepcopy(patch_seg)
        for _ in range(self.repetitions):
            # start_time = time.time()
            patch_img, patch_seg, blurred_blend_patch = self.force_touch_particles(patch_img, patch_seg, instance_seg, patch_negative, name)
            # print("Time: ", time.time() - start_time)
        # sitk.WriteImage(sitk.GetImageFromArray(patch_img), join(save_dir, "{}_image_after.nii.gz".format(id)))
        # sitk.WriteImage(sitk.GetImageFromArray(patch_seg), join(save_dir, "{}_seg_after.nii.gz".format(id)))
        # diff_img = patch_img - patch_img_before
        # diff_seg = patch_seg - patch_seg_before
        # diff_img[diff_img == -1] = 0
        # sum_img = np.sum(diff_img != 0)
        # sum_seg = np.sum(diff_seg > 0)
        # sitk.WriteImage(sitk.GetImageFromArray(diff_img), join(save_dir, "{}_img_diff_{}.nii.gz".format(id, sum_img)))
        # sitk.WriteImage(sitk.GetImageFromArray(diff_seg), join(save_dir, "{}_seg_diff_{}.nii.gz".format(id, sum_seg)))
        # if blurred_blend_patch is not None:
        #     sitk.WriteImage(sitk.GetImageFromArray(blurred_blend_patch), join(save_dir, "{}_blend_{}.nii.gz".format(id, sum_seg)))
        return patch_img, patch_seg

    def force_touch_particles(self, patch1_img, patch1_seg, patch1_instance_seg, patch1_negative, name):
        props1 = {prop.label: prop.bbox for prop in regionprops(patch1_instance_seg)}
        labels1 = list(props1.keys())
        if -1 in labels1:
            labels1.remove(-1)
        if not labels1:
            return patch1_img, patch1_seg, None
        label1 = random.choice(labels1)

        bbox1 = props1[label1]
        bbox1 = [[bbox1[i], bbox1[i + len(bbox1) // 2]] for i in range(len(bbox1) // 2)]
        bbox1 = np.asarray(bbox1)
        particle1_instance_seg = patch1_instance_seg[slicer(patch1_instance_seg, bbox1)]
        # sitk.WriteImage(sitk.GetImageFromArray(patch1_img), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/{}_patch1_img.nii.gz".format(name))
        # sitk.WriteImage(sitk.GetImageFromArray(patch1_seg), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/{}_patch1_seg.nii.gz".format(name))
        # sitk.WriteImage(sitk.GetImageFromArray(patch1_instance_seg), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/{}_patch1_instance_seg.nii.gz".format(name))
        # sitk.WriteImage(sitk.GetImageFromArray(particle1_instance_seg), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/{}_particle1_instance_seg.nii.gz".format(name))

        particle2_img, particle2_instance_seg, border_core2_seg, label2 = self.load_rand_particle(name)

        if particle2_img is None:
            return patch1_img, patch1_seg, None
        # sitk.WriteImage(sitk.GetImageFromArray(particle2_img), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/{}_particle2_img.nii.gz".format(name))
        # sitk.WriteImage(sitk.GetImageFromArray(border_core2_seg), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/{}_border_core2_seg.nii.gz".format(name))
        # sitk.WriteImage(sitk.GetImageFromArray(patch2_instance_seg), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/{}_patch2_instance_seg.nii.gz".format(name))
        # sitk.WriteImage(sitk.GetImageFromArray(particle2_instance_seg), "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp/{}_particle2_instance_seg.nii.gz".format(name))

        offset = bbox1[:, 0]
        move_vector = self.comp_move_vector(particle1_instance_seg == label1, particle2_instance_seg == label2)

        move_vector = np.rint(move_vector * 0.85).astype(np.int32)  # V3: 0.85, V4: 1.15

        ini = offset + move_vector
        fin = offset + move_vector + np.asarray(particle2_instance_seg.shape)
        indices = np.stack([ini, fin], axis=-1)


        crop_indices = [[None, None], [None, None], [None, None]]
        requires_crop = False
        for axis in range(len(patch1_img.shape)):
            if indices[axis, 0] < 0:
                crop_indices[axis][0] = indices[axis, 0] * -1
                indices[axis, 0] = 0
                requires_crop = True
            if patch1_img.shape[axis] < indices[axis, 1]:
                crop_indices[axis][1] = patch1_img.shape[axis] - indices[axis, 1]
                indices[axis, 1] = patch1_img.shape[axis]
                requires_crop = True
        if requires_crop:
            particle2_img = particle2_img[slicer(particle2_img, crop_indices)]
            particle2_instance_seg = particle2_instance_seg[slicer(particle2_instance_seg, crop_indices)]
            border_core2_seg = border_core2_seg[slicer(border_core2_seg, crop_indices)]

        bbox_size = indices[:, 1] - indices[:, 0]
        if (bbox_size <= 0).any():
            return patch1_img, patch1_seg, None

        blend_particle = (particle2_instance_seg == label2)
        blend_patch = np.zeros_like(patch1_seg, dtype=np.float32)
        blend_patch[slicer(blend_patch, indices)] = blend_particle
        blurred_blend_patch = gaussian(blend_patch, sigma=1)  # 0.75
        patch2_img = np.zeros_like(patch1_img)
        patch2_img[slicer(patch2_img, indices)] = particle2_img
        patch1_img = patch1_img * (1 - blurred_blend_patch) + patch2_img * blurred_blend_patch
        patch2_seg = np.zeros_like(patch1_seg)
        patch2_seg[slicer(patch2_seg, indices)] = border_core2_seg
        patch1_seg = patch1_seg * (1 - blend_patch) + patch2_seg * blend_patch
        patch1_img[patch1_negative == 1] = -1
        patch1_seg[patch1_negative == 1] = -1

        # np.putmask(patch1_img[slicer(patch1_img, indices)], particle2_instance_seg == label2, particle2_img)
        # patch1_img[patch1_negative == 1] = -1
        # np.putmask(patch1_seg[slicer(patch1_seg, indices)], particle2_instance_seg == label2, border_core2_seg)
        # patch1_seg[patch1_negative == 1] = -1

        return patch1_img, patch1_seg, blurred_blend_patch

    def load_rand_particle(self, name):
        labels = self.metadata[name]
        if not labels:
            return None, None, None, None
        label = random.choice(list(labels.keys()))
        bbox = labels[label]
        # bbox = [[bbox[i], bbox[i + len(bbox) // 2]] for i in range(len(bbox) // 2)]
        # bbox = np.asarray(bbox)
        img = np.load(join(self.preprocessed_data_dir, "nnUNetData_plans_v2.1_stage1", name + ".npy"), mmap_mode='r')[0]
        instance_seg = zarr.open(join(self.raw_data_dir, self.instance_seg_folder, name + ".zarr"), mode='r')["arr_0"].astype(np.uint16)
        border_core_seg = zarr.open(join(self.raw_data_dir, self.border_core_seg_folder, name + ".zarr"), mode='r')["arr_0"]
        img = img[slicer(img, bbox)]
        instance_seg = instance_seg[slicer(instance_seg, bbox)]
        border_core_seg = border_core_seg[slicer(border_core_seg, bbox)]
        img = np.array(img)
        # save_dir = "/home/k539i/Documents/datasets/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Task310_particle_seg/tmp2"
        # id = str(uuid.uuid4())[:8]
        # sitk.WriteImage(sitk.GetImageFromArray(border_core_seg), join(save_dir, "{}_{}_border_core.nii.gz".format(name, id)))
        # sitk.WriteImage(sitk.GetImageFromArray(instance_seg), join(save_dir, "{}_{}_instance.nii.gz".format(name, id)))
        return img, instance_seg, border_core_seg, label

    # def get_contour(self, instance_seg, label):
    #     mask = instance_seg == label
    #     eroded_mask = erosion(mask)
    #     contour = mask - eroded_mask
    #     return contour


    def comp_move_vector(self, particle1_seg, particle2_seg):
        indices1 = np.nonzero(particle1_seg)
        pos1 = np.argmax(indices1[0])
        pos1 = (indices1[0][pos1], indices1[1][pos1], indices1[2][pos1])
        pos1 = np.array(pos1)

        indices2 = np.nonzero(particle2_seg)
        pos2 = np.argmin(indices2[0])
        pos2 = (indices2[0][pos2], indices2[1][pos2], indices2[2][pos2])
        pos2 = np.array(pos2)

        move_vector = pos1 - pos2

        return move_vector