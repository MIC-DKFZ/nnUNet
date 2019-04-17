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

import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.tensor_utilities import flip
from torch import nn
import torch
from scipy.ndimage.filters import gaussian_filter


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        self.input_shape_must_be_divisible_by = None
        self.conv_op = None
        self.num_classes = None
        super(NeuralNetwork, self).__init__()
        self.inference_apply_nonlin = lambda x:x

    def predict_3D(self, x, do_mirroring, num_repeats=1, use_train_mode=False, batch_size=1, mirror_axes=(0, 1, 2),
                   tiled=False, tile_in_z=True, step=2, patch_size=None, regions_class_order=None, use_gaussian=False,
                   pad_border_mode="edge", pad_kwargs=None):
        """
        :param x: (c, x, y , z)
        :param do_mirroring:
        :param num_repeats:
        :param use_train_mode:
        :param batch_size:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param patch_size:
        :param regions_class_order:
        :param use_gaussian:
        :return:
        """
        print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)
        if len(mirror_axes) > 0 and max(mirror_axes) > 2:
            raise ValueError("mirror axes. duh")
        current_mode = self.training
        if use_train_mode is not None and use_train_mode:
            self.train()
        elif use_train_mode is not None and not use_train_mode:
            self.eval()
        else:
            pass
        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"
        if self.conv_op == nn.Conv3d:
            if tiled:
                res = self._internal_predict_3D_3Dconv_tiled(x, num_repeats, batch_size, tile_in_z, step, do_mirroring,
                                                             mirror_axes, patch_size, regions_class_order, use_gaussian,
                                                             pad_border_mode, pad_kwargs=pad_kwargs)
            else:
                res = self._internal_predict_3D_3Dconv(x, do_mirroring, num_repeats, patch_size, batch_size,
                                                       mirror_axes, regions_class_order, pad_border_mode, pad_kwargs=pad_kwargs)
        elif self.conv_op == nn.Conv2d:
            if tiled:
                res = self._internal_predict_3D_2Dconv_tiled(x, do_mirroring, num_repeats, batch_size, mirror_axes,
                                                             step, patch_size, regions_class_order, use_gaussian,
                                                             pad_border_mode, pad_kwargs=pad_kwargs)
            else:
                res = self._internal_predict_3D_2Dconv(x, do_mirroring, num_repeats, patch_size, batch_size,
                                                       mirror_axes, regions_class_order, pad_border_mode, pad_kwargs=pad_kwargs)
        else:
            raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")
        if use_train_mode is not None:
            self.train(current_mode)
        return res

    def predict_2D(self, x, do_mirroring, num_repeats=1, use_train_mode=False, batch_size=1, mirror_axes=(0, 1),
                   tiled=False, step=2, patch_size=None, regions_class_order=None, use_gaussian=False,
                   pad_border_mode="edge", pad_kwargs=None):
        if len(mirror_axes) > 0 and max(mirror_axes) > 1:
            raise ValueError("mirror axes. duh")
        assert len(x.shape) == 3, "data must have shape (c,x,y)"
        current_mode = self.training
        if use_train_mode is not None and use_train_mode:
            self.train()
        elif use_train_mode is not None and not use_train_mode:
            self.eval()
        else:
            pass
        if self.conv_op == nn.Conv3d:
            raise RuntimeError("Cannot predict 2d if the network is 3d. Dummy.")
        elif self.conv_op == nn.Conv2d:
            if tiled:
                res = self._internal_predict_2D_2Dconv_tiled(x, num_repeats, batch_size, step, do_mirroring,
                                                             mirror_axes, patch_size, regions_class_order,
                                                             use_gaussian, pad_border_mode, pad_kwargs=pad_kwargs)
            else:
                res = self._internal_predict_2D_2Dconv(x, do_mirroring, num_repeats, None, batch_size, mirror_axes,
                                                       regions_class_order, pad_border_mode, pad_kwargs=pad_kwargs)
        else:
            raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")
        if use_train_mode is not None:
            self.train(current_mode)
        return res

    def _internal_maybe_mirror_and_pred_3D(self, x, num_repeats, mirror_axes, do_mirroring=True, mult=None):
        with torch.no_grad():
            x_torch = torch.from_numpy(x).float()
            if self.get_device() == "cpu":
                x_torch = x_torch.cpu()
            else:
                x_torch = x_torch.cuda(self.get_device())

            result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:])).float()
            if self.get_device() == "cpu":
                result_torch = result_torch.cpu()
            else:
                result_torch = result_torch.cuda(self.get_device())

            num_results = num_repeats
            if do_mirroring:
                mirror_idx = 8
                num_results *= 2 ** len(mirror_axes)
            else:
                mirror_idx = 1

            for i in range(num_repeats):
                for m in range(mirror_idx):
                    if m == 0:
                        pred = self.inference_apply_nonlin(self(x_torch))
                        result_torch += 1/num_results * pred

                    if m == 1 and (2 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(x_torch, 4)))
                        result_torch += 1/num_results * flip(pred, 4)

                    if m == 2 and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(x_torch, 3)))
                        result_torch += 1/num_results * flip(pred, 3)

                    if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(x_torch, 4), 3)))
                        result_torch += 1/num_results * flip(flip(pred, 4), 3)

                    if m == 4 and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(x_torch, 2)))
                        result_torch += 1/num_results * flip(pred, 2)

                    if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(x_torch, 4), 2)))
                        result_torch += 1/num_results * flip(flip(pred, 4), 2)

                    if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(x_torch, 3), 2)))
                        result_torch += 1/num_results * flip(flip(pred, 3), 2)

                    if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(flip(x_torch, 3), 2), 4)))
                        result_torch += 1/num_results * flip(flip(flip(pred, 3), 2), 4)

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch.detach().cpu().numpy()

    def _internal_predict_3D_3Dconv_tiled(self, x, num_repeats, BATCH_SIZE=None, tile_in_z=True, step=2,
                                          do_mirroring=True, mirror_axes=(0, 1, 2), patch_size=None,
                                          regions_class_order=None, use_gaussian=False, pad_border_mode="edge", pad_kwargs=None):
        "x must be (c, x, y, z)"
        assert len(x.shape) == 4, "x must be (c, x, y, z)"
        with torch.no_grad():
            assert patch_size is not None, "patch_size cannot be None for tiled prediction"

            data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)

            data = data[None]

            if BATCH_SIZE is not None:
                data = np.vstack([data] * BATCH_SIZE)

            input_size = [1, x.shape[0]] + list(patch_size)
            if not tile_in_z:
                input_size[2] = data.shape[2]
                patch_size[0] = data.shape[2]
            input_size = [int(i) for i in input_size]

            a = torch.rand(input_size).float()
            if self.get_device() == "cpu":
                a = a.cpu()
            else:
                a = a.cuda(self.get_device())

            # dummy run to see number of classes
            nb_of_classes = self(a).size()[1]

            result = np.zeros([nb_of_classes] + list(data.shape[2:]), dtype=np.float32)
            result_numsamples = np.zeros([nb_of_classes] + list(data.shape[2:]), dtype=np.float32)
            if use_gaussian:
                tmp = np.zeros(patch_size)
                center_coords = [i//2 for i in patch_size]
                sigmas = [i//8 for i in patch_size]
                tmp[tuple(center_coords)] = 1
                tmp_smooth = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
                tmp_smooth = tmp_smooth / tmp_smooth.max() * 1
                add = tmp_smooth + 1e-8
            else:
                add = np.ones(patch_size)

            if self.get_device() == "cpu":
                add_torch = torch.from_numpy(add).cpu().float()
            else:
                add_torch = torch.from_numpy(add).cuda(self.get_device()).float()

            data_shape = data.shape
            center_coord_start = np.array([i//2 for i in patch_size]).astype(int)
            center_coord_end = np.array([data_shape[i + 2] - patch_size[i] // 2 for i in range(len(patch_size))]).astype(int)
            num_steps = np.ceil([(center_coord_end[i] - center_coord_start[i]) / (patch_size[i] / step) for i in range(3)])
            step_size = np.array([(center_coord_end[i] - center_coord_start[i]) / (num_steps[i] + 1e-8) for i in range(3)])
            step_size[step_size == 0] = 9999999
            xsteps = np.round(np.arange(center_coord_start[0], center_coord_end[0]+1e-8, step_size[0])).astype(int)
            ysteps = np.round(np.arange(center_coord_start[1], center_coord_end[1]+1e-8, step_size[1])).astype(int)
            zsteps = np.round(np.arange(center_coord_start[2], center_coord_end[2]+1e-8, step_size[2])).astype(int)

            for x in xsteps:
                lb_x = x - patch_size[0] // 2
                ub_x = x + patch_size[0] // 2
                for y in ysteps:
                    lb_y = y - patch_size[1] // 2
                    ub_y = y + patch_size[1] // 2
                    for z in zsteps:
                        lb_z = z - patch_size[2] // 2
                        ub_z = z + patch_size[2] // 2
                        result[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += \
                            self._internal_maybe_mirror_and_pred_3D(data[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z],
                                                                          num_repeats, mirror_axes, do_mirroring, add_torch)[0]
                        result_numsamples[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add

            slicer = tuple([slice(0, result.shape[i]) for i in range(len(result.shape) - (len(slicer) - 1))] + slicer[1:])
            result = result[slicer]
            result_numsamples = result_numsamples[slicer]

            softmax_pred = result / result_numsamples

            # patient_data = patient_data[:, :old_shape[0], :old_shape[1], :old_shape[2]]
            if regions_class_order is None:
                predicted_segmentation = softmax_pred.argmax(0)
            else:
                predicted_segmentation_shp = softmax_pred[0].shape
                predicted_segmentation = np.zeros(predicted_segmentation_shp, dtype=np.float32)
                for i, c in enumerate(regions_class_order):
                    predicted_segmentation[softmax_pred[i] > 0.5] = c
        return predicted_segmentation, None, softmax_pred, None

    def _internal_predict_2D_2Dconv(self, x, do_mirroring, num_repeats, min_size=None, BATCH_SIZE=None,
                                    mirror_axes=(0, 1), regions_class_order=None, pad_border_mode="edge", pad_kwargs=None):
        with torch.no_grad():
            _ = None
            #x, old_shape = pad_patient_2D_incl_c(x, self.input_shape_must_be_divisible_by, min_size)
            x, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True, self.input_shape_must_be_divisible_by)
            """pad_res = []
            for i in range(x.shape[0]):
                t, old_shape = pad_patient_2D(x[i], self.input_shape_must_be_divisible_by, None)
                pad_res.append(t[None])

            x = np.vstack(pad_res)"""

            new_shp = x.shape

            data = np.zeros(tuple([1] + list(new_shp)), dtype=np.float32)

            data[0] = x

            if BATCH_SIZE is not None:
                data = np.vstack([data] * BATCH_SIZE)

            result = self._internal_maybe_mirror_and_pred_2D(data, num_repeats, mirror_axes, do_mirroring)[0]

            slicer = tuple([slice(0, result.shape[i]) for i in range(len(result.shape) - (len(slicer) - 1))] + slicer[1:])
            result = result[slicer]
            softmax_pred = result

            if regions_class_order is None:
                predicted_segmentation = softmax_pred.argmax(0)
            else:
                predicted_segmentation_shp = softmax_pred[0].shape
                predicted_segmentation = np.zeros(predicted_segmentation_shp, dtype=np.float32)
                for i, c in enumerate(regions_class_order):
                    predicted_segmentation[softmax_pred[i] > 0.5] = c
        return predicted_segmentation, _, softmax_pred, _

    def _internal_predict_3D_3Dconv(self, x, do_mirroring, num_repeats, min_size=None, BATCH_SIZE=None,
                                    mirror_axes=(0, 1, 2), regions_class_order=None, pad_border_mode="edge",
                                    pad_kwargs=None):
        with torch.no_grad():
            x, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True, self.input_shape_must_be_divisible_by)
            #x, old_shape = pad_patient_3D_incl_c(x, self.input_shape_must_be_divisible_by, min_size)

            new_shp = x.shape

            data = np.zeros(tuple([1] + list(new_shp)), dtype=np.float32)

            data[0] = x

            if BATCH_SIZE is not None:
                data = np.vstack([data] * BATCH_SIZE)

            stacked = self._internal_maybe_mirror_and_pred_3D(data, num_repeats, mirror_axes, do_mirroring, None)[0]

            slicer = tuple([slice(0, stacked.shape[i]) for i in range(len(stacked.shape) - (len(slicer) - 1))] + slicer[1:])
            stacked = stacked[slicer]
            softmax_pred = stacked

            if regions_class_order is None:
                predicted_segmentation = softmax_pred.argmax(0)
            else:
                predicted_segmentation_shp = softmax_pred[0].shape
                predicted_segmentation = np.zeros(predicted_segmentation_shp, dtype=np.float32)
                for i, c in enumerate(regions_class_order):
                    predicted_segmentation[softmax_pred[i] > 0.5] = c
        return predicted_segmentation, None, softmax_pred, None

    def _internal_maybe_mirror_and_pred_2D(self, x, num_repeats, mirror_axes, do_mirroring=True, mult=None):
        with torch.no_grad():
            x_torch = torch.from_numpy(x).float()
            if self.get_device() == "cpu":
                x_torch = x_torch.cpu()
            else:
                x_torch = x_torch.cuda(self.get_device())

            result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:])).float()
            if self.get_device() == "cpu":
                result_torch = result_torch.cpu()
            else:
                result_torch = result_torch.cuda(self.get_device())

            num_results = num_repeats
            if do_mirroring:
                mirror_idx = 4
                num_results *= 2 ** len(mirror_axes)
            else:
                mirror_idx = 1

            for i in range(num_repeats):
                for m in range(mirror_idx):
                    if m == 0:
                        pred = self.inference_apply_nonlin(self(x_torch))
                        result_torch += 1/num_results * pred

                    if m == 1 and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(x_torch, 3)))
                        result_torch += 1/num_results * flip(pred, 3)

                    if m == 2 and (0 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(x_torch, 2)))
                        result_torch += 1/num_results * flip(pred, 2)

                    if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                        pred = self.inference_apply_nonlin(self(flip(flip(x_torch, 3), 2)))
                        result_torch += 1/num_results * flip(flip(pred, 3), 2)

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch.detach().cpu().numpy()

    def _internal_predict_2D_2Dconv_tiled(self, patient_data, num_repeats, BATCH_SIZE=None, step=2,
                                     do_mirroring=True, mirror_axes=(0, 1), patch_size=None, regions_class_order=None,
                                          use_gaussian=False, pad_border_mode="edge", pad_kwargs=None):
        with torch.no_grad():
            tile_size = patch_size
            assert tile_size is not None, "patch_size cannot be None for tiled prediction"
            # pad images so that their size is a multiple of tile_size
            data, slicer = pad_nd_image(patient_data, tile_size, pad_border_mode, pad_kwargs, True)

            data = data[None]

            if BATCH_SIZE is not None:
                data = np.vstack([data] * BATCH_SIZE)

            input_size = [1, patient_data.shape[0]] + list(tile_size)
            input_size = [int(i) for i in input_size]
            a = torch.rand(input_size).float()
            if self.get_device() == "cpu":
                a = a.cpu()
            else:
                a = a.cuda(self.get_device())

            # dummy run to see number of classes
            nb_of_classes = self(a).size()[1]

            result = np.zeros([nb_of_classes] + list(data.shape[2:]), dtype=np.float32)
            result_numsamples = np.zeros([nb_of_classes] + list(data.shape[2:]), dtype=np.float32)
            if use_gaussian:
                tmp = np.zeros(tile_size, dtype=np.float32)
                center_coords = [i//2 for i in tile_size]
                sigmas = [i // 8 for i in tile_size]
                tmp[tuple(center_coords)] = 1
                tmp_smooth = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
                tmp_smooth = tmp_smooth / tmp_smooth.max() * 1
                add = tmp_smooth
            else:
                add = np.ones(tile_size)

            if self.get_device() == "cpu":
                add_torch = torch.from_numpy(add).cpu().float()
            else:
                add_torch = torch.from_numpy(add).cuda(self.get_device()).float()

            data_shape = data.shape
            center_coord_start = np.array([i//2 for i in patch_size]).astype(int)
            center_coord_end = np.array([data_shape[i + 2] - patch_size[i] // 2 for i in range(len(patch_size))]).astype(int)
            num_steps = np.ceil([(center_coord_end[i] - center_coord_start[i]) / (patch_size[i] / step) for i in range(2)])
            step_size = np.array([(center_coord_end[i] - center_coord_start[i]) / (num_steps[i] + 1e-8) for i in range(2)])
            step_size[step_size == 0] = 9999999
            xsteps = np.round(np.arange(center_coord_start[0], center_coord_end[0]+1e-8, step_size[0])).astype(int)
            ysteps = np.round(np.arange(center_coord_start[1], center_coord_end[1]+1e-8, step_size[1])).astype(int)

            for x in xsteps:
                lb_x = x - patch_size[0] // 2
                ub_x = x + patch_size[0] // 2
                for y in ysteps:
                    lb_y = y - patch_size[1] // 2
                    ub_y = y + patch_size[1] // 2
                    result[:, lb_x:ub_x, lb_y:ub_y] += \
                        self._internal_maybe_mirror_and_pred_2D(data[:, :, lb_x:ub_x, lb_y:ub_y],
                                                                num_repeats, mirror_axes, do_mirroring, add_torch)[0]
                    result_numsamples[:, lb_x:ub_x, lb_y:ub_y] += add

            slicer = tuple([slice(0, result.shape[i]) for i in range(len(result.shape) - (len(slicer) - 1))] + slicer[1:])
            result = result[slicer]
            result_numsamples = result_numsamples[slicer]
            softmax_pred = result / result_numsamples

            if regions_class_order is None:
                predicted_segmentation = softmax_pred.argmax(0)
            else:
                predicted_segmentation_shp = softmax_pred[0].shape
                predicted_segmentation = np.zeros(predicted_segmentation_shp, dtype=np.float32)
                for i, c in enumerate(regions_class_order):
                    predicted_segmentation[softmax_pred[i] > 0.5] = c
        return predicted_segmentation, None, softmax_pred, None

    def _internal_predict_3D_2Dconv(self, data, do_mirroring, num_repeats, min_size=None, BATCH_SIZE=None,
                                    mirror_axes=(0, 1), regions_class_order=None, pad_border_mode="edge", pad_kwargs=None):
        assert len(data.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        for s in range(data.shape[1]):
            pred_seg, bayesian_predictions, softmax_pres, uncertainty = \
                self._internal_predict_2D_2Dconv(data[:, s], do_mirroring, num_repeats, min_size, BATCH_SIZE,
                                                 mirror_axes, regions_class_order, pad_border_mode, pad_kwargs)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, None, softmax_pred, None

    def predict_3D_pseudo3D_2Dconv(self, data, do_mirroring, num_repeats, min_size=None, BATCH_SIZE=None,
                                   mirror_axes=(0, 1), regions_class_order=None, pseudo3D_slices=5):
        assert len(data.shape) == 4, "data must be c, x, y, z"
        assert pseudo3D_slices % 2 == 1, "pseudo3D_slices must be odd"
        extra_slices = (pseudo3D_slices - 1) // 2
        shp_for_pad = np.array(data.shape)
        shp_for_pad[1] = extra_slices
        pad = np.zeros(shp_for_pad, dtype=np.float32)
        data = np.concatenate((pad, data, pad), 1)
        predicted_segmentation = []
        softmax_pred = []
        for s in range(extra_slices, data.shape[1] - extra_slices):
            d = data[:, (s - extra_slices):(s + extra_slices + 1)]
            d = d.reshape((-1, d.shape[-2], d.shape[-1]))
            pred_seg, bayesian_predictions, softmax_pres, uncertainty = \
                self._internal_predict_2D_2Dconv(d, do_mirroring, num_repeats, min_size, BATCH_SIZE, mirror_axes,
                                                 regions_class_order)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, None, softmax_pred, None

    def _internal_predict_3D_2Dconv_tiled(self, data, do_mirroring, num_repeats, BATCH_SIZE=None, mirror_axes=(0, 1),
                                          step=2, patch_size=None, regions_class_order=None, use_gaussian=False,
                                          pad_border_mode="edge", pad_kwargs=None):
        assert len(data.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        for s in range(data.shape[1]):
            pred_seg, bayesian_predictions, softmax_pres, uncertainty = \
                self._internal_predict_2D_2Dconv_tiled(data[:, s], num_repeats, BATCH_SIZE, step, do_mirroring,
                                                       mirror_axes, patch_size, regions_class_order, use_gaussian,
                                                       pad_border_mode=pad_border_mode, pad_kwargs=pad_kwargs)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, None, softmax_pred, None



