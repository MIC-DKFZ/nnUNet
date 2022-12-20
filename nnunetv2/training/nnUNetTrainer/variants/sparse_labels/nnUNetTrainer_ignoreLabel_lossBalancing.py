import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss_ignlossbalancing(nn.Module):
    def __init__(self, ce_kwargs, ignore_label=None, batch_dice=False, ddp= False, do_bg=False):
        """
        balance the contribution of samples depending on how much in them is annotated (=not ignore label)
        """
        super(DC_and_CE_loss_ignlossbalancing, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label
        self.batch_dice = batch_dice
        self.ddp = ddp
        self.do_bg = do_bg
        self.ce = RobustCrossEntropyLoss(**ce_kwargs, reduction='none')

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # dice
        with torch.no_grad():
            if self.ignore_label is not None:
                non_ignore_mask = (target != self.ignore_label).float()
                target_dice = torch.clone(target)
                target_dice[target == self.ignore_label] = 0
            else:
                non_ignore_mask = torch.ones_like(target, dtype=torch.float)
                target_dice = target
            shp_x = net_output.shape
            axes = list(range(2, len(shp_x)))
            sample_fg_percent = non_ignore_mask.sum((1, *axes)) / np.prod(shp_x[2:])
        x = softmax_helper_dim1(net_output)
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, target_dice, axes, non_ignore_mask, False)

        if self.ddp:
            raise NotImplementedError('No DDP for you!')  # we would need to synchronize sample_weights and ce losses per sample as well

        sample_weights = sample_fg_percent / torch.clip(sample_fg_percent.sum(), min=1e-8) * shp_x[0]
        tp *= sample_weights[:, None]
        fp *= sample_weights[:, None]
        fn *= sample_weights[:, None]

        if self.batch_dice:
            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + 1e-5) / (denominator + 1e-5)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = - dc.mean()

        # ce
        # ce is just 0 where the ignore label is. Careful in aggregation!
        ce_loss = self.ce(net_output, target[:, 0].long())
        ce_loss = ((ce_loss.sum([i - 1 for i in axes]) / torch.clip(non_ignore_mask[:, 0].sum([i - 1 for i in axes]), min=1e-8)) * sample_weights).mean()

        result = ce_loss + dc
        return result


class nnUNetTrainer_ignoreLabel_lossBalancing(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError('this dont work for region based training')
        else:
            loss = DC_and_CE_loss_ignlossbalancing({}, self.label_manager.ignore_label,
                                                   self.configuration_manager.batch_dice, self.is_ddp, False)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
