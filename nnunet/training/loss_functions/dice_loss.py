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

import torch
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., apply_nonlin=None, batch_dice=False, do_bg=True, smooth_in_nom=True,
                 background_weight=1, rebalance_weights=None, square_nominator=False, square_denom=False):
        """
        hahaa no documentation for you today
        :param smooth:
        :param apply_nonlin:
        :param batch_dice:
        :param do_bg:
        :param smooth_in_nom:
        :param background_weight:
        :param rebalance_weights:
        """
        super(SoftDiceLoss, self).__init__()
        self.square_denom = square_denom
        self.square_nominator = square_nominator
        if not do_bg:
            assert background_weight == 1, "if there is no bg, then set background weight to 1 you dummy"
        self.rebalance_weights = rebalance_weights
        self.background_weight = background_weight
        if smooth_in_nom:
            self.smooth_in_nom = smooth
        else:
            self.smooth_in_nom = 0
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.y_onehot = None

    def forward(self, x, y):
        with torch.no_grad():
            y = y.long()
        shp_x = x.shape
        shp_y = y.shape
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        # now x and y should have shape (B, C, X, Y(, Z))) and (B, 1, X, Y(, Z))), respectively
        y_onehot = torch.zeros(shp_x)
        if x.device.type == "cuda":
            y_onehot = y_onehot.cuda(x.device.index)
        y_onehot.scatter_(1, y, 1)
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]
        if not self.batch_dice:
            if self.background_weight != 1 or (self.rebalance_weights is not None):
                raise NotImplementedError("nah son")
            l = soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom, self.square_nominator, self.square_denom)
        else:
            l = soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom,
                                      background_weight=self.background_weight,
                                      rebalance_weights=self.rebalance_weights)
        return l


def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1, rebalance_weights=None,
                        square_nominator=False, square_denom=False):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:] # this is the case when use_bg=False
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    tp = sum_tensor(net_output * gt, axes, keepdim=False)
    fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    weights = torch.ones(tp.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float()
        if net_output.device.type == "cuda":
            rebalance_weights = rebalance_weights.cuda(net_output.device.index)
        tp = tp * rebalance_weights
        fn = fn * rebalance_weights

    nominator = tp

    if square_nominator:
        nominator = nominator ** 2

    if square_denom:
        denom = 2 * tp ** 2 + fp ** 2 + fn ** 2
    else:
        denom = 2 * tp + fp + fn

    result = (- ((2 * nominator + smooth_in_nom) / (denom + smooth)) * weights).mean()
    return result


def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1., square_nominator=False, square_denom=False):
    axes = tuple(range(2, len(net_output.size())))
    if square_nominator:
        intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    else:
        intersect = sum_tensor((net_output * gt) ** 2, axes, keepdim=False)
    if square_denom:
        denom = sum_tensor(net_output ** 2 + gt ** 2, axes, keepdim=False)
    else:
        denom = sum_tensor(net_output + gt, axes, keepdim=False)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()
    return result


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result
