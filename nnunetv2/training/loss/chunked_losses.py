"""
Batch-chunked CE and Dice, for training with many classes (100+ classes at large patch sizes).

PyTorch's nll_loss2d CUDA kernel (cross entropy backward) indexes with int32. Once
batch * num_classes * voxels grows large enough the backward hits an illegal memory access
(observed on 3d_fullres, 192^3 patches, 118 classes at >= 3 samples per GPU: 3 * 118 * 192^3 > 2^31). The
einsum-free dice ops additionally allocate a (b, c, *spatial) one-hot / product tensor which is
wasteful at large batch sizes. Both are avoided by processing the batch in chunks small enough that
batch_chunk * num_classes * voxels stays under this many elements per kernel launch.

Below the limit nothing is chunked and the exact parent computation runs, so these classes are safe
drop-in replacements at any size.
"""
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.dice import global_sum

DEFAULT_MAX_ELEMENTS_PER_CHUNK = 2 ** 31


def _chunk_size(batch_size: int, elements_per_sample: int, max_elements_per_chunk: int) -> int:
    return min(batch_size, max(1, int(max_elements_per_chunk // max(1, elements_per_sample))))


class ChunkedRobustCrossEntropyLoss(RobustCrossEntropyLoss):
    """
    Same as RobustCrossEntropyLoss but computes the mean over the batch in chunks to keep every
    nll_loss2d kernel launch below the int32 indexing limit.

    For reduction='mean' with weight=None and label_smoothing=0 (the nnU-Net default) this equals
    the unchunked loss up to floating-point summation order: mean == sum(per-voxel losses) / count,
    and we accumulate the per-chunk 'sum' reductions and divide by the total valid voxel count.
    For any other configuration (or when no chunking is required) we fall back to the exact parent
    computation, which is then bit-for-bit identical.
    """
    max_elements_per_chunk = DEFAULT_MAX_ELEMENTS_PER_CHUNK

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        chunk = _chunk_size(input.shape[0], input[0].numel(), self.max_elements_per_chunk)
        if chunk >= input.shape[0] or self.reduction != 'mean' or self.label_smoothing != 0 \
                or self.weight is not None:
            # exact parent computation (also the small deep-supervision scales land here)
            return nn.CrossEntropyLoss.forward(self, input, target)

        loss_sum = input.new_zeros(())
        count = torch.zeros((), device=input.device, dtype=torch.long)
        for i in range(0, input.shape[0], chunk):
            inp = input[i:i + chunk]
            tgt = target[i:i + chunk]
            loss_sum = loss_sum + F.cross_entropy(inp, tgt, weight=self.weight,
                                                  ignore_index=self.ignore_index, reduction='sum')
            count = count + (tgt != self.ignore_index).sum()
        return loss_sum / count.clamp(min=1).to(loss_sum.dtype)


class ChunkedMemoryEfficientSoftDiceLoss(MemoryEfficientSoftDiceLoss):
    """
    Bit-for-bit identical to MemoryEfficientSoftDiceLoss, but the per-sample statistics
    (intersect, sum_pred, sum_gt) are computed one batch-chunk at a time and concatenated back into
    the full (b, c) tensors before the reduction tail. Because those statistics reduce only over the
    spatial axes they are independent across the batch, so chunking + concatenation reproduces the
    exact same tensors (and hence the exact same batch_dice sum(0), dice formula and mean) with no
    change in arithmetic ordering.
    """
    max_elements_per_chunk = DEFAULT_MAX_ELEMENTS_PER_CHUNK

    def forward(self, x, y, loss_mask=None):
        chunk = _chunk_size(x.shape[0], x[0].numel(), self.max_elements_per_chunk)
        if chunk >= x.shape[0]:
            # exact parent computation (also the small deep-supervision scales land here)
            return super().forward(x, y, loss_mask)

        axes = tuple(range(2, x.ndim))
        intersect_parts, sum_pred_parts, sum_gt_parts = [], [], []
        for i in range(0, x.shape[0], chunk):
            xc = x[i:i + chunk]
            yc = y[i:i + chunk]
            mc = loss_mask[i:i + chunk] if loss_mask is not None else None

            if self.apply_nonlin is not None:
                xc = self.apply_nonlin(xc)

            with torch.no_grad():
                if xc.ndim != yc.ndim:
                    yc = yc.view((yc.shape[0], 1, *yc.shape[1:]))
                if xc.shape == yc.shape:
                    # gt is already one hot
                    y_onehot = yc.to(torch.float32)
                else:
                    y_onehot = torch.zeros(xc.shape, device=xc.device, dtype=torch.float32)
                    y_onehot.scatter_(1, yc.long(), 1)
                if not self.do_bg:
                    y_onehot = y_onehot[:, 1:]
                sum_gt = y_onehot.sum(axes, dtype=torch.float32) if mc is None \
                    else (y_onehot * mc).sum(axes, dtype=torch.float32)

            # MUST be outside no_grad, otherwise no gradients
            xc_fg = xc[:, 1:] if not self.do_bg else xc
            if mc is None:
                intersect = (xc_fg * y_onehot).sum(axes, dtype=torch.float32)
                sum_pred = xc_fg.sum(axes, dtype=torch.float32)
            else:
                intersect = (xc_fg * y_onehot * mc).sum(axes, dtype=torch.float32)
                sum_pred = (xc_fg * mc).sum(axes, dtype=torch.float32)

            intersect_parts.append(intersect)
            sum_pred_parts.append(sum_pred)
            sum_gt_parts.append(sum_gt)

        intersect = torch.cat(intersect_parts, 0)
        sum_pred = torch.cat(sum_pred_parts, 0)
        sum_gt = torch.cat(sum_gt_parts, 0)

        # ---- identical tail to MemoryEfficientSoftDiceLoss ----
        # global_sum is exactly what MemoryEfficientSoftDiceLoss uses, so the two stay in lockstep.
        if self.batch_dice:
            if self.ddp:
                intersect = global_sum(intersect)
                sum_pred = global_sum(sum_pred)
                sum_gt = global_sum(sum_gt)

            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        dc = (2 * intersect + self.smooth) / (sum_gt + sum_pred + float(self.smooth)).clamp_min(1e-8)
        dc = dc.mean()
        return -dc


class DC_and_CE_loss_chunked(DC_and_CE_loss):
    """
    Drop-in replacement for DC_and_CE_loss that chunks both the cross entropy and the (memory
    efficient) dice over the batch dimension. Dice is bit-for-bit identical to the unchunked loss;
    cross entropy is identical up to floating-point summation order (see ChunkedRobustCrossEntropyLoss).
    """
    # The budget is also a class attribute, so that a
    # subclass (and through it a trainer) can lower it without touching _build_loss. Chunking below the
    # int32 limit is still exact, and it is the only way to use this to cut the dice's peak VRAM at all
    # at 160^3 / 118 classes, where batch * classes * voxels stays under 2**31 and nothing would be
    # chunked otherwise. The default is the int32 limit. NOTE the saving is small - ~10% of the loss' fwd+bwd peak,
    # zero off its forward peak - because chunking cannot shrink what autograd retains (the softmax
    # output and y_onehot, both full (b, c, *spatial) fp32, both kept whole no matter the chunk count).
    # See the comment above DC_and_CE_loss_chunked_lowMem in
    # nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainer_chunkedLoss.py for the measurements.
    max_elements_per_chunk = DEFAULT_MAX_ELEMENTS_PER_CHUNK

    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=ChunkedMemoryEfficientSoftDiceLoss,
                 max_elements_per_chunk: int = None):
        if max_elements_per_chunk is None:
            max_elements_per_chunk = type(self).max_elements_per_chunk
        # DC_and_CE_loss injects ignore_index into ce_kwargs and builds self.ce / self.dc
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label, dice_class)
        # swap in the chunked cross entropy (ce_kwargs now already carries ignore_index if set)
        self.ce = ChunkedRobustCrossEntropyLoss(**ce_kwargs)
        # propagate the budget to both components
        self.ce.max_elements_per_chunk = max_elements_per_chunk
        if hasattr(self.dc, 'max_elements_per_chunk'):
            self.dc.max_elements_per_chunk = max_elements_per_chunk
