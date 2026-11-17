"""PGPS (nnunetv2/training/nnUNetTrainer/variants/patch_size/nnUNetTrainer_PGPS.py): ladder, checkpoint_best, CUDA-graph
stages, and that its changes to PyTorch stay inside its own steps."""
import copy
import os
import unittest
from unittest import mock

import numpy as np
import torch
import torch._inductor.cudagraph_trees as cudagraph_trees
import torch.nn.functional as F

from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.patch_size import nnUNetTrainer_PGPS as pgps
from nnunetv2.training.nnUNetTrainer.variants.patch_size.nnUNetTrainer_PGPS import (
    PGPSEfficiencyMixin,
    nnUNetTrainer_DASegOrd0_NoMirroring_PGPS_2700epochs_SV,
    nnUNetTrainer_DASegOrd0_NoMirroring_PGPS_5400epochs_SV,
    nnUNetTrainer_PGPS,
    nnUNetTrainer_PGPS_2700epochs,
    single_voxel_instance_norm,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

PATCH = [192, 192, 192]
STRIDES = [[1, 1, 1]] + [[2, 2, 2]] * 5
CUDA = torch.cuda.is_available()


def _fake_base_init(device: str = 'cpu', num_epochs: int = 1000, batch_size: int = 2):
    # stands in for nnUNetTrainer.__init__: what the stage logic, checkpoint_best and train_step read
    def init(self, plans, configuration, fold, dataset_json, device_=None):
        self.configuration_manager = ConfigurationManager(
            {'patch_size': list(PATCH), 'batch_size': batch_size, 'architecture': {'arch_kwargs': {'strides': STRIDES}}})
        self.device = torch.device(device)
        self.is_ddp = False
        self.world_size = 1
        self.global_rank = 0
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.print_to_log_file = lambda *args, **kwargs: None
        self.logger = MetaLogger(None, False)
        self._best_ema = None
        self.output_folder = 'fold_0'
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.was_initialized = False
    return init


def _make(device: str = 'cpu', num_epochs: int = 5400, trainer_class=nnUNetTrainer_PGPS, batch_size: int = 2):
    with mock.patch.object(nnUNetTrainer, '__init__', _fake_base_init(device, 1000, batch_size)):
        t = trainer_class({}, '3d_fullres', 0, {})
    t.num_epochs = num_epochs
    return t


def _stage_starts(t):
    return [e for e in range(1, t.num_epochs) if t.patch_size_for_epoch(e) != t.patch_size_for_epoch(e - 1)]


class TestLadder(unittest.TestCase):
    def test_ladder_192(self):
        ladder = _make().pgps_ladder()
        self.assertEqual(len(ladder), 16)
        self.assertEqual(ladder[:4], [(32, 32, 32), (64, 32, 32), (64, 64, 32), (64, 64, 64)])
        self.assertEqual(ladder[-1], tuple(PATCH))

    def test_equal_epochs_per_stage(self):
        t = _make(num_epochs=160)
        self.assertEqual(_stage_starts(t), list(range(10, 160, 10)))

    def test_recipe_trainers(self):
        for cls, epochs in ((nnUNetTrainer_PGPS, 1000), (nnUNetTrainer_PGPS_2700epochs, 2700),
                            (nnUNetTrainer_DASegOrd0_NoMirroring_PGPS_2700epochs_SV, 2700),
                            (nnUNetTrainer_DASegOrd0_NoMirroring_PGPS_5400epochs_SV, 5400)):
            with self.subTest(cls=cls.__name__), mock.patch.object(nnUNetTrainer, '__init__', _fake_base_init()):
                t = cls({}, '3d_fullres', 0, {})
                self.assertEqual(t.num_epochs, epochs)
                self.assertEqual(len(t.pgps_ladder()), 16)
                mro = type(t).__mro__
                self.assertLess(mro.index(PGPSEfficiencyMixin), mro.index(nnUNetTrainer))


# ------------------------------------------------------------------------------------------------- checkpoint_best ----
NUM_EPOCHS = 160      # 16 stages at 192^3 -> 10 epochs per stage, the final stage is epochs 150-159
FINAL = 150


def _make_cb(num_epochs: int = NUM_EPOCHS):
    t = _make(num_epochs=num_epochs)
    t.writes = []
    t.save_checkpoint = lambda filename: t.writes.append((t.current_epoch, os.path.basename(filename)))
    return t


def _train(t, ema: dict, epochs):
    """What run_training does per epoch, reduced to the stage switch and the checkpoint_best decision."""
    for e in epochs:
        t.current_epoch = e
        t._maybe_update_patch_size(rebuild_dataloaders=False)
        t.logger.log('ema_fg_dice', ema[e], e)
        t._maybe_save_checkpoint_best()
    return [e for e, f in t.writes if f == 'checkpoint_best.safetensors']


def _resume(t):
    """A --c run: a new trainer that loads current_epoch, the logger and _best_ema from the checkpoint."""
    r = _make_cb(t.num_epochs)
    r.logger.load_checkpoint(copy.deepcopy(t.logger.get_checkpoint()))
    r._best_ema = t._best_ema
    r.current_epoch = t.current_epoch + 1
    return r


class TestCheckpointBest(unittest.TestCase):
    def test_final_stage_boundary(self):
        t = _make_cb()
        self.assertEqual(t.first_epoch_of_final_stage(), FINAL)
        self.assertNotEqual(t.patch_size_for_epoch(FINAL - 1), tuple(PATCH))
        self.assertEqual(t.patch_size_for_epoch(FINAL), tuple(PATCH))

    def test_writes_only_in_final_stage(self):
        ema = {e: e / NUM_EPOCHS for e in range(NUM_EPOCHS)}
        self.assertEqual(_train(_make_cb(), ema, range(NUM_EPOCHS)), list(range(FINAL, NUM_EPOCHS)))

    def test_earlier_stage_best_does_not_block_final_stage(self):
        ema = {e: 0.9 for e in range(FINAL)}
        ema.update(dict(zip(range(FINAL, NUM_EPOCHS), [0.5, 0.7, 0.6, 0.65, 0.8, 0.75, 0.8, 0.81, 0.3, 0.2])))
        self.assertEqual(_train(_make_cb(), ema, range(NUM_EPOCHS)), [150, 151, 154, 157])

    def test_resume_inside_final_stage_keeps_best(self):
        ema = {e: e / NUM_EPOCHS for e in range(FINAL)}
        ema.update(dict(zip(range(FINAL, NUM_EPOCHS), [0.5, 0.7, 0.6, 0.65, 0.68, 0.69, 0.71, 0.6, 0.6, 0.6])))
        t = _make_cb()
        self.assertEqual(_train(t, ema, range(FINAL + 5)), [150, 151])
        self.assertEqual(_train(_resume(t), ema, range(FINAL + 5, NUM_EPOCHS)), [156])

    def test_resume_into_final_stage(self):
        ema = {e: e / NUM_EPOCHS for e in range(FINAL)}
        ema.update({e: 0.1 + e / 1000 for e in range(FINAL, NUM_EPOCHS)})
        t = _make_cb()
        self.assertEqual(_train(t, ema, range(FINAL)), [])
        self.assertEqual(_train(_resume(t), ema, range(FINAL, NUM_EPOCHS)), list(range(FINAL, NUM_EPOCHS)))

    def test_run_too_short_for_the_full_patch_size(self):
        # 8 epochs over 16 stages: the last epoch is in stage 15 of 16, which then counts as final
        t = _make_cb(num_epochs=8)
        self.assertNotEqual(t.patch_size_for_epoch(7), tuple(PATCH))
        self.assertEqual(t.first_epoch_of_final_stage(), 7)
        self.assertEqual(_train(t, {e: e / 10 for e in range(8)}, range(8)), [7])


# ------------------------------------------------------------------------------------------ stages and CUDA graphs ----
def _walk_stages(t, compiled: bool = True, cudnn_benchmark: bool = False, events: list = None):
    """
    Patch sizes of the stages that train with CUDA graphs, and the epochs at which all compiled code was dropped.
    events (optional) collects (what, epoch) for zero_grad, reset and the cuDNN pre-search, in call order.
    """
    stages, resets = [], []
    events = [] if events is None else events

    def reset():
        resets.append(t.current_epoch)
        events.append(('reset', t.current_epoch))
    t.optimizer = mock.Mock()
    t.optimizer.zero_grad = lambda set_to_none: events.append(('zero_grad' if set_to_none else 'zero_grad(keep)',
                                                               t.current_epoch))
    benchmark_before = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
    try:
        with mock.patch.object(torch._dynamo, 'reset', reset), \
                mock.patch.object(pgps, 'empty_cache', lambda device: None), \
                mock.patch.object(torch.cuda, 'memory_reserved', lambda device: 0), \
                mock.patch.object(nnUNetTrainer, '_do_i_compile', lambda self: compiled), \
                mock.patch.object(PGPSEfficiencyMixin, '_cudnn_presearch',
                                  lambda self: events.append(('presearch', self.current_epoch))):
            for e in range(t.num_epochs):
                t.current_epoch = e
                t._maybe_update_patch_size(rebuild_dataloaders=False)
                if t._cudagraphs_this_stage and t.current_stage_patch_size not in stages:
                    stages.append(t.current_stage_patch_size)
    finally:
        torch.backends.cudnn.benchmark = benchmark_before
    return stages, resets


class TestStages(unittest.TestCase):
    def test_cudagraph_stages_and_a_reset_at_every_switch(self):
        # 1 GPU, bs 2: CUDA graphs up to 96x64x64. All compiled code is dropped at every stage switch, so each stage
        # compiles its own static shapes
        t = _make('cuda')
        stages, resets = _walk_stages(t)
        self.assertEqual(stages, [(32, 32, 32), (64, 32, 32), (64, 64, 32), (64, 64, 64), (96, 64, 64)])
        self.assertEqual(resets, _stage_starts(t))

    def test_no_reset_on_entering_the_first_stage_of_a_resumed_run(self):
        # a --c run starts in a later stage; nothing has been compiled yet in that process
        t = _make('cuda')
        starts = _stage_starts(t)
        t.current_epoch = starts[5] + 3          # starts[5] is where stage 6 begins
        events = []
        with mock.patch.object(torch._dynamo, 'reset', lambda: events.append('reset')), \
                mock.patch.object(nnUNetTrainer, '_do_i_compile', lambda self: True):
            t.optimizer = mock.Mock()
            t._maybe_update_patch_size(rebuild_dataloaders=False)
        self.assertEqual(events, [])
        self.assertEqual(t.current_stage_patch_size, t.pgps_ladder()[6])

    def test_gradients_dropped_before_every_reset(self):
        events = []
        _walk_stages(_make('cuda'), events=events)
        for i, (what, epoch) in enumerate(events):
            if what == 'reset':
                self.assertEqual(events[i - 1], ('zero_grad', epoch))

    def test_cudnn_presearch_once_per_cudagraph_stage(self):
        t = _make('cuda')
        events = []
        _walk_stages(t, cudnn_benchmark=True, events=events)
        self.assertEqual([e for what, e in events if what == 'presearch'], [0] + _stage_starts(t)[:4])
        # after the reset of the stage it belongs to
        self.assertLess(events.index(('reset', _stage_starts(t)[0])), events.index(('presearch', _stage_starts(t)[0])))
        events = []
        _walk_stages(_make('cuda'), cudnn_benchmark=False, events=events)
        self.assertNotIn('presearch', [what for what, _ in events])

    def test_patch_voxels_decide(self):
        # the patch alone decides, not the batch
        self.assertEqual(_walk_stages(_make('cuda', batch_size=8)), _walk_stages(_make('cuda')))

    def test_off(self):
        # no compile (CPU, or compile disabled): no CUDA graphs and no resets
        for t, compiled in ((_make('cpu'), False), (_make('cuda'), False)):
            self.assertEqual(_walk_stages(t, compiled), ([], []))
        t = _make('cuda')
        t.pgps_cudagraphs = False
        stages, resets = _walk_stages(t)
        self.assertEqual(stages, [])
        self.assertEqual(resets, _stage_starts(t))


# ------------------------------------------------------------------------------- PyTorch changes stay inside PGPS ----
class _RecordingSteps(nnUNetTrainer):
    """Records which spatial size check is installed while the base steps run."""
    def train_step(self, batch: dict) -> dict:
        self.seen.append(('train', F._verify_spatial_size))
        return {'loss': np.full((), 0.5)}

    def validation_step(self, batch: dict) -> dict:
        self.seen.append(('val', F._verify_spatial_size))
        return {}


class _PGPSOnRecordingSteps(PGPSEfficiencyMixin, _RecordingSteps):
    pass


class TestScope(unittest.TestCase):
    def test_single_voxel_bottleneck_in_the_first_stage_only(self):
        t = _make('cpu', num_epochs=16)
        flags = []
        for e in range(16):
            t.current_epoch = e
            t._maybe_update_patch_size(rebuild_dataloaders=False)
            flags.append(t._single_voxel_bottleneck)
        self.assertEqual(flags, [True] + [False] * 15)

    def test_check_skipped_only_inside_the_steps_of_that_stage(self):
        original = F._verify_spatial_size
        t = _make('cpu', num_epochs=16, trainer_class=_PGPSOnRecordingSteps)
        t.seen = []
        for e in (0, 1):
            t.current_epoch = e
            t._maybe_update_patch_size(rebuild_dataloaders=False)
            t.train_step({})
            t.validation_step({})
            self.assertIs(F._verify_spatial_size, original)
        self.assertEqual(t.seen, [('train', pgps._skip_spatial_size_check), ('val', pgps._skip_spatial_size_check),
                                  ('train', original), ('val', original)])

    def test_restored_after_an_exception(self):
        original = F._verify_spatial_size
        with self.assertRaises(RuntimeError):
            with single_voxel_instance_norm(True):
                self.assertIs(F._verify_spatial_size, pgps._skip_spatial_size_check)
                raise RuntimeError
        self.assertIs(F._verify_spatial_size, original)
        with single_voxel_instance_norm(False):
            self.assertIs(F._verify_spatial_size, original)

    def test_instance_norm_inside_and_outside(self):
        norm = torch.nn.InstanceNorm3d(4, affine=True).train()
        x = torch.randn(2, 4, 1, 1, 1)
        with self.assertRaises(ValueError):
            norm(x)
        with single_voxel_instance_norm(True):
            norm(x)
        with self.assertRaises(ValueError):
            norm(x)

    def test_initialize_changes_no_dynamo_setting(self):
        from torch._dynamo import config as dynamo_config
        names = [n for n in ('automatic_dynamic_shapes', 'recompile_limit', 'cache_size_limit') if hasattr(dynamo_config, n)]
        before = {n: getattr(dynamo_config, n) for n in names}
        t = _make('cpu')
        with mock.patch.object(nnUNetTrainer, 'initialize', lambda self: None):
            t.initialize()
        self.assertEqual({n: getattr(dynamo_config, n) for n in names}, before)


# ------------------------------------------------------------------------------------------------ other pieces ----
class TestCudnnPresearch(unittest.TestCase):
    def test_training_does_not_see_it(self):
        # parameters, buffers (BatchNorm running stats), gradients, the global RNG and the train / eval mode unchanged
        t = _make('cpu')
        torch.manual_seed(0)
        t.network = torch.nn.Sequential(torch.nn.Conv3d(2, 4, 3, padding=1), torch.nn.BatchNorm3d(4),
                                        torch.nn.Conv3d(4, 3, 1))
        t.network.eval()
        t.batch_size, t.num_input_channels, t.current_stage_patch_size = 2, 2, (8, 8, 8)
        state = copy.deepcopy(t.network.state_dict())
        rng = torch.get_rng_state()
        t._cudnn_presearch()
        for k, v in t.network.state_dict().items():
            self.assertTrue(torch.equal(v, state[k]), k)
        self.assertTrue(all(p.grad is None for p in t.network.parameters()))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(t.network.training)


class _OwnTrainStep(nnUNetTrainer):
    def train_step(self, batch: dict) -> dict:
        return {'loss': np.full((), 0.5)}


class _PGPSOnOwnTrainStep(PGPSEfficiencyMixin, _OwnTrainStep):
    pass


class TestTrainStep(unittest.TestCase):
    def test_keeps_a_train_step_override(self):
        # the mixin only wraps train_step, so a trainer between it and nnUNetTrainer keeps its own
        t = _make(trainer_class=_PGPSOnOwnTrainStep)
        self.assertEqual(float(t.train_step({})['loss']), 0.5)


def _small_net():
    # two stride-2 convolutions: an 4^3 input reaches the InstanceNorm with a single voxel
    return torch.nn.Sequential(torch.nn.Conv3d(1, 8, 3, padding=1), torch.nn.InstanceNorm3d(8, affine=True),
                               torch.nn.LeakyReLU(), torch.nn.Conv3d(8, 8, 3, stride=2, padding=1),
                               torch.nn.Conv3d(8, 8, 3, stride=2, padding=1), torch.nn.InstanceNorm3d(8, affine=True),
                               torch.nn.LeakyReLU(), torch.nn.Upsample(scale_factor=4), torch.nn.Conv3d(8, 2, 1))


@unittest.skipUnless(CUDA, 'needs CUDA')
class TestCompiled(unittest.TestCase):
    def _steps(self, cudagraphs: bool, single_voxel: bool, state, batch, k: int = 3):
        t = _make('cuda')
        net = _small_net().cuda()
        net.load_state_dict(state)
        t.network = torch.compile(net)
        t.optimizer, _ = t.configure_optimizers()
        t.grad_scaler = torch.amp.GradScaler('cuda')
        # compiled, as _build_loss does: then the loss is a CUDA-graph output too
        t.loss = torch.compile(lambda output, target: torch.nn.functional.cross_entropy(output.float(), target))
        t._cudagraphs_this_stage = cudagraphs
        t._single_voxel_bottleneck = single_voxel
        recorded, record = [], cudagraph_trees.cudagraphify_impl
        frames_before = torch._dynamo.utils.counters['stats']['unique_graphs']
        with mock.patch.object(cudagraph_trees, 'cudagraphify_impl',
                               lambda *a, **kw: recorded.append(1) or record(*a, **kw)):
            outputs = [t.train_step(batch) for _ in range(k)]
        graphs = torch._dynamo.utils.counters['stats']['unique_graphs'] - frames_before
        losses = [float(o['loss']) for o in outputs]
        self.assertEqual(len(recorded) > 0, cudagraphs)
        return [p.detach().clone() for p in net.parameters()], losses, graphs

    def _state_and_batch(self, size: int):
        torch.manual_seed(0)
        state = copy.deepcopy(_small_net().cuda().state_dict())
        batch = {'data': torch.randn(2, 1, size, size, size), 'target': torch.randint(0, 2, (2, size, size, size))}
        return state, batch

    def test_cudagraphs_bit_identical(self):
        deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        try:
            state, batch = self._state_and_batch(16)
            torch._dynamo.reset()
            p_plain, l_plain, _ = self._steps(False, False, state, batch)
            torch._dynamo.reset()
            p_cg, l_cg, _ = self._steps(True, False, state, batch)
            torch._dynamo.reset()
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark
        self.assertEqual(l_plain, l_cg)
        for a, b in zip(p_plain, p_cg):
            self.assertTrue(torch.equal(a, b))

    def test_single_voxel_bottleneck_compiles_once(self):
        # the skipped check is one function object, so torch.compile does not recompile at every step
        state, batch = self._state_and_batch(4)
        torch._dynamo.reset()
        _, losses, graphs = self._steps(True, True, state, batch)
        torch._dynamo.reset()
        self.assertTrue(all(np.isfinite(losses)))
        self.assertLessEqual(graphs, 2)       # the network and the loss
        self.assertIsNot(F._verify_spatial_size, pgps._skip_spatial_size_check)


if __name__ == '__main__':
    unittest.main()
