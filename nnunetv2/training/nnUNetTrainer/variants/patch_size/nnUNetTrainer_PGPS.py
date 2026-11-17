"""
Progressive growing of patch size (PGPS-Efficiency) from Fischer et al., 'Progressive Growing of Patch Size: Curriculum
Learning for Accelerated and Improved Medical Image Segmentation' (arXiv:2510.23241). Reference implementation:
https://github.com/StefanFischer/nnUNet-ProgressiveGrowingOfPatchSize

Training starts at one patch_size_divisor unit per axis (the total stride of the encoder, e.g. 32^3 for a network with six
stages) and grows the axis with the lowest current value by one unit at a time until the configured patch size is reached.
Every stage gets the same number of epochs, so a run ends with the configured patch size, which is the one used for
inference. Crops are cropped smaller, not rescaled. Batch size, iterations per epoch and the learning rate schedule are
unchanged. At 192^3 the ladder has 16 stages, at 160^3 13.

CUDA graphs for the small stages: in the first stages an iteration is bound by the host, because issuing the kernels takes
longer than the GPU work. With torch.compile on CUDA, the training graphs of stages whose patch has at most
pgps_cudagraph_max_patch_voxels voxels are recorded as CUDA graphs (inductor's triton.cudagraphs, set around train_step only,
so the validation graphs get none). Numerics are the same as without. On entering such a stage, cuDNN's algorithm search runs
once outside the graphs (_cudnn_presearch): in the CUDA-graph warm-up its workspace would stay in the graph's memory pool.

What PGPS does to PyTorch is confined to its own steps:
- At every stage switch all compiled code is dropped (torch._dynamo.reset()). No stage is visited twice, so nothing reusable
  is lost; each stage compiles its own static shapes, and no global dynamo setting has to change.
- In training mode, InstanceNorm refuses an input with a single spatial element, which the bottleneck receives in a stage with
  one unit per axis. Like the reference implementation, the trainer skips that check
  (torch.nn.functional._verify_spatial_size), but only around the steps of such a stage, and restores it right after.

plans.json keeps the configured patch size: only this trainer's configuration manager holds the stage patch size. A --val or
--c run starts from the configured patch size and derives the stage from the epoch counter.
"""
import contextlib
from typing import Tuple

import numpy as np
import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.validation.nnUNetTrainer_streamingValidation import \
    nnUNetTrainer_DASegOrd0_NoMirroring_SV
from nnunetv2.utilities.helpers import dummy_context, empty_cache


def _skip_spatial_size_check(size) -> None:
    # One function object for the whole process: torch.compile guards on the identity of the function it traced, so a new
    # object per step would recompile every step.
    return None


@contextlib.contextmanager
def single_voxel_instance_norm(allowed: bool):
    """Within this context (if allowed), InstanceNorm accepts an input with a single spatial element in training mode."""
    if not allowed:
        yield
        return
    original = F._verify_spatial_size
    F._verify_spatial_size = _skip_spatial_size_check
    try:
        yield
    finally:
        F._verify_spatial_size = original


class PGPSEfficiencyMixin:
    """
    Mix in first, so that its hooks come before the trainer's in the MRO:
        class MyTrainer_PGPS(PGPSEfficiencyMixin, MyTrainer): ...
    """
    # one patch_size_divisor unit per axis at the start, as in the paper
    pgps_min_units_per_axis = 1
    # CUDA graphs for the training steps of stages whose patch has at most this many voxels (see the module docstring)
    pgps_cudagraphs = True
    pgps_cudagraph_max_patch_voxels = 96 * 64 * 64

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # the configured patch size, for which the network is built
        self.full_patch_size = tuple(self.configuration_manager.patch_size)
        self.current_stage_patch_size = None
        # the split does not change between stages: do not read it again for every new set of dataloaders
        self._split_cache = None
        self._pgps_ladder_cache = None
        self._pgps_compile = None
        self._cudagraphs_this_stage = False
        self._single_voxel_bottleneck = False

    @property
    def patch_size_divisor(self) -> Tuple[int, ...]:
        """The smallest patch size increment per axis: the total stride of the encoder."""
        divisor = np.prod(self.configuration_manager.pool_op_kernel_sizes, axis=0, dtype=np.int64)
        return tuple(int(i) for i in divisor)

    def pgps_ladder(self) -> list:
        """
        Every patch size the schedule visits, smallest first, ending with the configured one. Starts at
        pgps_min_units_per_axis divisor units per axis; each step grows the axis with the lowest current value by its
        divisor. Ties go to the lower axis index.
        """
        if self._pgps_ladder_cache is None:
            full, div = self.full_patch_size, self.patch_size_divisor
            for f, d in zip(full, div):
                assert f % d == 0, f"The configured patch size {full} is not divisible by {div}. PGPS cannot build a ladder."
            current = [min(f, d * self.pgps_min_units_per_axis) for f, d in zip(full, div)]
            ladder = [tuple(current)]
            while tuple(current) != tuple(full):
                axis = min((i for i in range(len(current)) if current[i] < full[i]), key=lambda i: (current[i], i))
                current[axis] += div[axis]
                ladder.append(tuple(current))
            self._pgps_ladder_cache = ladder
        return self._pgps_ladder_cache

    def patch_size_for_epoch(self, epoch: int) -> tuple:
        ladder = self.pgps_ladder()
        # the same number of epochs per stage; the configured patch size is the last stage
        return ladder[min(len(ladder) - 1, int(epoch * len(ladder) / self.num_epochs))]

    def is_final_stage(self, epoch: int) -> bool:
        # the stage of the last epoch: the configured patch size, unless the run is too short to get there
        return self.patch_size_for_epoch(epoch) == self.patch_size_for_epoch(self.num_epochs - 1)

    def first_epoch_of_final_stage(self) -> int:
        return next(e for e in range(self.num_epochs) if self.is_final_stage(e))

    def initialize(self):
        assert tuple(self.configuration_manager.patch_size) == self.full_patch_size, (
            f"The network must be built for the configured patch size {self.full_patch_size}, "
            f"got {tuple(self.configuration_manager.patch_size)}.")
        super().initialize()
        ladder = self.pgps_ladder()
        self.print_to_log_file(f"PGPS ladder ({len(ladder)} stages, {self.num_epochs // len(ladder)} epochs each): "
                               + " -> ".join("x".join(str(i) for i in p) for p in ladder))
        self.print_to_log_file(f"checkpoint_best: final stage only (epochs {self.first_epoch_of_final_stage()} - "
                               f"{self.num_epochs - 1})")

    def on_train_start(self):
        if not self.was_initialized:
            # the network is built for the configured patch size, so initialize before the first stage is set
            self.initialize()
        # the stage patch size goes in before super() creates the dataloaders. On a --c resume current_epoch comes from
        # the checkpoint, so this picks the stage the run stopped in
        self._maybe_update_patch_size(rebuild_dataloaders=False)
        with single_voxel_instance_norm(self._single_voxel_bottleneck):
            super().on_train_start()

    def on_epoch_start(self):
        super().on_epoch_start()
        self._maybe_update_patch_size(rebuild_dataloaders=True)

    def _compiles(self) -> bool:
        if self._pgps_compile is None:
            self._pgps_compile = self._do_i_compile()
        return self._pgps_compile

    def _maybe_update_patch_size(self, rebuild_dataloaders: bool):
        patch_size = self.patch_size_for_epoch(self.current_epoch)
        if patch_size == self.current_stage_patch_size:
            return
        first_stage_of_this_process = self.current_stage_patch_size is None
        was_cudagraphs = self._cudagraphs_this_stage
        self.current_stage_patch_size = patch_size
        self.configuration_manager.configuration["patch_size"] = list(patch_size)
        compiles = self._compiles()
        if compiles and not first_stage_of_this_process:
            # Drop all compiled code: no stage is visited twice, and the next one compiles its own static shapes. The
            # parameters' .grad of a CUDA-graph step live in the graph's memory pool, which the reset frees: drop them
            # first, anything that touched them after the reset would read freed memory
            self.optimizer.zero_grad(set_to_none=True)
            torch._dynamo.reset()
            empty_cache(self.device)
        self._cudagraphs_this_stage = (self.pgps_cudagraphs and self.device.type == 'cuda' and compiles and
                                       int(np.prod(patch_size)) <= self.pgps_cudagraph_max_patch_voxels)
        self._single_voxel_bottleneck = all(p == d for p, d in zip(patch_size, self.patch_size_divisor))
        if was_cudagraphs and not self._cudagraphs_this_stage:
            self.print_to_log_file(f'CUDA graph stages done, reserved now '
                                   f'{torch.cuda.memory_reserved(self.device) / 2 ** 30:.2f} GiB')
        fraction = float(np.prod(patch_size) / np.prod(self.full_patch_size))
        self.print_to_log_file(f"Patch size: {patch_size} of {self.full_patch_size} ({fraction:.1%} of the volume)"
                               + (", CUDA graphs" if self._cudagraphs_this_stage else ""))
        if self._cudagraphs_this_stage and torch.backends.cudnn.enabled and torch.backends.cudnn.benchmark:
            self._cudnn_presearch()
        if rebuild_dataloaders:
            self._restart_dataloaders()
        if self.is_final_stage(self.current_epoch):
            # only the final stage competes for checkpoint_best (_maybe_save_checkpoint_best). Its best so far comes from
            # the logged EMA of the final stage's epochs, so a --c resume inside the final stage keeps it
            ema = self.logger.get_value('ema_fg_dice', step=None)[self.first_epoch_of_final_stage():self.current_epoch]
            self._best_ema = max(ema) if len(ema) > 0 else None

    def _cudnn_presearch(self):
        """
        cuDNN's algorithm search (cudnn.benchmark) for the convolutions of a new CUDA-graph stage, run eagerly before its
        first compiled step. Left to the compiled step, the search happens in the CUDA-graph warm-up, which allocates from
        the graph's private memory pool with PyTorch's cache emptying switched off: the search workspace, up to several GB
        per convolution, then stays in the pool until the graphs are dropped. Here it is freed right away, and ATen's
        algorithm cache serves the compiled graph's convolutions (same shapes, dtypes and layouts).

        One forward and backward pass of the uncompiled network (under DDP the inner module: no collectives) at the stage's
        batch and patch size. The input comes from its own generator, so the global RNG state is untouched; gradients are
        dropped and buffers restored, so training does not see it.
        """
        net = self.network.module if self.is_ddp else self.network
        net = getattr(net, '_orig_mod', net)
        was_training = net.training
        net.train()
        buffers = {k: v.clone() for k, v in net.named_buffers()}
        for p in net.parameters():
            p.grad = None
        generator = torch.Generator(device=self.device).manual_seed(0)
        x = torch.randn((self.batch_size, self.num_input_channels, *self.current_stage_patch_size), device=self.device,
                        generator=generator)
        with single_voxel_instance_norm(self._single_voxel_bottleneck), \
                (autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context()):
            output = net(x)
            loss = sum(o.float().mean() for o in (output if isinstance(output, (list, tuple)) else [output]))
        loss.backward()
        for p in net.parameters():
            p.grad = None
        with torch.no_grad():
            for k, v in net.named_buffers():
                v.copy_(buffers[k])
        net.train(was_training)
        del x, output, loss, buffers
        empty_cache(self.device)

    def _maybe_save_checkpoint_best(self):
        # Before the final stage the pseudo Dice comes from smaller validation patches, so only the final stage competes
        if self.is_final_stage(self.current_epoch):
            super()._maybe_save_checkpoint_best()

    def train_step(self, batch: dict) -> dict:
        # inductor reads the flag when it compiles a graph, and the training graphs are compiled here
        with inductor_config.patch({'triton.cudagraphs': self._cudagraphs_this_stage}), \
                single_voxel_instance_norm(self._single_voxel_bottleneck):
            outputs = super().train_step(batch)
        if self._cudagraphs_this_stage:
            # the loss lives in the CUDA graph's memory, which the next replay overwrites; it is read at the epoch's end
            outputs['loss'] = outputs['loss'].clone()
        return outputs

    def validation_step(self, batch: dict) -> dict:
        with single_voxel_instance_norm(self._single_voxel_bottleneck):
            return super().validation_step(batch)

    def do_split(self):
        # called by get_dataloaders() at every stage switch; the split is fixed for the run
        if self._split_cache is None:
            self._split_cache = super().do_split()
        return self._split_cache

    def _restart_dataloaders(self):
        """The patch size is part of the augmentation pipeline, so the dataloaders are built again."""
        self._shutdown_dataloaders()
        self.dataloader_train = self.dataloader_val = None
        empty_cache(self.device)
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()


class nnUNetTrainer_PGPS(PGPSEfficiencyMixin, nnUNetTrainer):
    """nnUNetTrainer with PGPS, 1000 epochs: about 0.4x the compute of the dense 1000 epochs."""


class nnUNetTrainer_PGPS_2700epochs(nnUNetTrainer_PGPS):
    """nnUNetTrainer with PGPS, 2700 epochs: about the compute of the dense 1000 epochs."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2700


class nnUNetTrainer_DASegOrd0_NoMirroring_PGPS_2700epochs_SV(PGPSEfficiencyMixin, nnUNetTrainer_DASegOrd0_NoMirroring_SV):
    """
    Many classes (see nnUNetTrainer_DASegOrd0_NoMirroring_SV) with PGPS, 2700 epochs: about the compute of the dense 1000
    epochs.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2700


class nnUNetTrainer_DASegOrd0_NoMirroring_PGPS_5400epochs_SV(PGPSEfficiencyMixin, nnUNetTrainer_DASegOrd0_NoMirroring_SV):
    """
    Many classes (see nnUNetTrainer_DASegOrd0_NoMirroring_SV) with PGPS, 5400 epochs: about the compute of the dense 2000
    epochs (nnUNetTrainer_DASegOrd0_NoMirroring_2000epochs_SV).
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5400
