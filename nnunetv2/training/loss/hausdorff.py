import torch
from torch import nn
from torch.nn import functional as F
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, SoftDiceLossSquared, DC_and_CE_loss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
import numpy as np
import time as time

import math
from sklearn.utils.extmath import cartesian
from sklearn.metrics.pairwise import pairwise_distances

import cv2 as cv
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

torch.set_default_dtype(torch.float32)

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
Copied from: https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py
"""

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class BinaryHausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(BinaryHausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # Was commented out before
        pred = softmax_helper(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss

class HausdorffERLoss(BinaryHausdorffERLoss):
    """Multiclass Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__(alpha, erosions)

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = torch.from_numpy(cross * 0.2)
        self.kernel3D = torch.from_numpy(np.array([bound, cross, bound]) * (1 / 7))

    # @torch.no_grad()
    def perform_erosion(self, pred: torch.Tensor, target: torch.Tensor, debug) -> torch.Tensor:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            convolve = F.conv3d
            kernel = self.kernel3D
        elif bound.ndim == 4:
            convolve = F.conv2d
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = torch.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            if debug: erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):
                breakpoint()
                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, stride=1)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if torch.max(erosion) > torch.min(erosion):
                    erosion = (erosion - torch.min(erosion)) / (torch.max(erosion)-torch.min(erosion))

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug: erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, c, x, y, z) or (b, c, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        start = time.time()
        # Was commented out before
        pred = softmax_helper(pred)
        
        shp_x = pred.shape
        shp_y = target.shape
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(pred.shape, target.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = target
            else:
                target = target.long()
                y_onehot = torch.zeros(shp_x)
                if pred.device.type == "cuda":
                    y_onehot = y_onehot.cuda(pred.device.index)
                y_onehot.scatter_(1, target, 1).float()

        loss = 0
        for c in range(shp_x[1]):
            eroted = self.perform_erosion(pred[:,c,...], y_onehot[:,c,...], debug)

            loss += eroted.mean()

        end = time.time()
        print(f"Hausdorff ER time: {end-start} seconds")

        return loss/shp_x[1]

class DC_CE_and_HausdorffER_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1, weight_ahd=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        :param weight_whd:
        """
        super().__init__(soft_dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                         log_dice, ignore_label)

        self.hd = HausdorffERLoss()
        self.weight_hd = weight_ahd

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        hd_loss = self.hd(net_output, target)

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_hd * hd_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result