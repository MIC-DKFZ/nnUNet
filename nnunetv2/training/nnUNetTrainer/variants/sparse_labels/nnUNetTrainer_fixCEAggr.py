import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class DC_and_CE_loss_fixCEAggr(DC_and_CE_loss):
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            if self.ignore_label is not None:
                non_ignore_mask = (target != self.ignore_label).float()
                has_fg = non_ignore_mask.sum() > 0
            else:
                non_ignore_mask = torch.ones_like(target, dtype=torch.float)
                has_fg = True

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
        else:
            target_dice = target

        dc_loss = self.dc(net_output, target_dice, loss_mask=non_ignore_mask) \
            if self.weight_dice != 0 else 0

        if has_fg:
            axes = list(range(2, len(net_output.shape)))
            ce_loss = self.ce(net_output, target[:, 0].long())
            ce_loss = (
                    ce_loss.sum([i - 1 for i in axes]) / torch.clip(non_ignore_mask[:, 0].sum([i - 1 for i in axes]),
                                                                    min=1e-8)
            ).mean()
        else:
            ce_loss = 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class nnUNetTrainer_ignoreLabel_fixCEAggr(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError
        else:
            loss = DC_and_CE_loss_fixCEAggr({'batch_dice': self.configuration_manager.batch_dice,
                                             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'reduction': 'none'}, weight_ce=1,
                                            weight_dice=1,
                                            ignore_label=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
