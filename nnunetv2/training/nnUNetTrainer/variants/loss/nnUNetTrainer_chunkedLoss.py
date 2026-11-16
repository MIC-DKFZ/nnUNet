"""
Trainers whose CE and Dice are chunked over the batch dimension, for datasets with many classes.

PyTorch's nll_loss2d CUDA kernel (cross entropy backward) indexes with int32, so the backward hits an
illegal memory access once batch * num_classes * voxels >= 2**31 - observed on 3d_fullres, 192^3 patches,
118 classes, at >= 3 samples per GPU. Below that limit nothing is chunked and the exact stock computation
runs, so these trainers are safe to use at any size.
"""
import torch

from nnunetv2.training.loss.chunked_losses import ChunkedMemoryEfficientSoftDiceLoss, DC_and_CE_loss_chunked
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class ChunkedLossMixin:
    """
    Swaps the loss classes nnUNetTrainer._build_loss instantiates for their batch-chunked counterparts.
    Implemented as class attributes rather than a _build_loss override so that it composes with any other
    trainer variant without per-combination code.
    """
    dice_class = ChunkedMemoryEfficientSoftDiceLoss
    dc_and_ce_class = DC_and_CE_loss_chunked


class nnUNetTrainer_chunkedLoss(ChunkedLossMixin, nnUNetTrainer):
    """Stock trainer, chunked loss."""


# --------------------------------------------------------------------------------------------------
# If the above OOM: the default budget (2**31) is the int32 crash limit, and at 160^3 / 118 classes /
# bs2 we sit below it, so nothing gets chunked. Lowering the budget forces chunking, which is still
# exact - but do not expect much. MEASURED at b=2 / 119 classes / 160^3 / AMP, full DC_and_CE,
# fwd+bwd peak: 18.23 GiB unchunked -> 16.42 GiB at 2 chunks. 1.8 GiB, i.e. ~10%, and the peak of the
# forward pass alone does not move at all.
#
# The reason it is only that much: chunking cannot touch what the autograd graph retains, and at this
# size that is 12.77 of the 18.23 GiB. Walking the graph and attributing every saved tensor to the node
# that saved it, at b=2 / 119 classes / 160^3 (one full fp32 tensor = 3.63 GiB):
#
#   3.63  SoftmaxBackward0._saved_result      softmax output                     <- dice
#   3.63  MulBackward0._saved_other           y_onehot                           <- dice
#   3.63  NllLoss2DBackward0._saved_self      log_softmax output, cast to fp32   <- CE
#   1.82  LogSoftmaxBackward0._saved_result   log_softmax output, fp16           <- CE
#   0.06  NllLoss2DBackward0._saved_target    target, int64                      <- CE
#
# The dice keeps 2 x full fp32: its softmax output (softmax saves its own output) and y_onehot - which is
# built under no_grad but is then saved as the OTHER operand of the mul in
# `intersect = (x * y_onehot).sum(axes)`, because that mul has to produce grad_x = grad_out * y_onehot.
# requires_grad=False does not stop a tensor from being saved that way. CE keeps its log_softmax output
# TWICE, once fp16 and once cast to fp32 for nll_loss2d.
#
# Chunking just builds each of these as n slices that all stay alive to the end of the loop - verified by
# sweeping the chunk count over 1/4/8 on the dice, where the retained total is CONSTANT at exactly 2x full
# fp32. At 2 chunks the total in fact goes UP slightly (12.77 -> 12.83 GiB) because the int64 target copy
# is made once per chunk. All chunking shrinks is the transient (x * y_onehot) product, freed right after
# its .sum() - hence one chunk's worth, once. It is also not monotonic in the chunk count (8 chunks
# measured WORSE than 4, 2.86 vs 2.67 GiB on a b=8 dice), from allocator fragmentation plus the torch.cat
# tail holding the parts and the result at the same time.
#
# So: worth trying as a last resort, not a fix. What DID work was none of this - it was widening the
# torch.compile scope in nnUNetTrainer._build_loss from loss.dc to the whole compound loss, which is
# where the real -35% lives. See the comment there.
#
# Two levers that look obvious here and were measured and REJECTED, so do not spend the time again:
#   - gather + scatter_add for intersect, so no one-hot exists at all (its backward needs only the
#     (b, 1, *patch_size) index tensor). Slower AND higher peak than the stock dice under compile,
#     because inductor already lowers the one-hot to a bool mask and fuses the mul+sum, while
#     scatter_add is atomics-bound and unfusable: 7.49 GiB / 26.8 ms vs 5.64 GiB / 13.1 ms at
#     b=2 / 119 / 128^3 (eager it is far worse still, 11.16 GiB / 87.2 ms).
#   - a custom autograd.Function with an analytic backward. Gradients come out bit-identical and
#     saved-for-backward drops to 0.96 GiB, but torch.compile cannot see inside a Function, so peak
#     rose to 8.42 GiB and it ran ~2x slower than compiled stock.
# torch.utils.checkpoint around the dice would still trade recompute for both dice rows; untested.
# --------------------------------------------------------------------------------------------------
class DC_and_CE_loss_chunked_lowMem(DC_and_CE_loss_chunked):
    """
    Chunk at 2**28 elements, i.e. one sample per chunk at 118 classes / 160^3. Buys ~10% off the
    loss' fwd+bwd peak and nothing off its forward peak; see the comment above before reaching for it.
    """
    max_elements_per_chunk = 2 ** 28


class LowMemChunkedLossMixin(ChunkedLossMixin):
    dc_and_ce_class = DC_and_CE_loss_chunked_lowMem


class nnUNetTrainer_chunkedLossLowMem(LowMemChunkedLossMixin, nnUNetTrainer):
    """Stock trainer, chunked loss with a lowered chunk budget."""


class nnUNetTrainer_chunkedLoss_5epochs(nnUNetTrainer_chunkedLoss):
    """used for debugging plans / checking that a many-class dataset fits and runs at all"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
