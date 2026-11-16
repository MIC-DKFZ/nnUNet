import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as trainer_module
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs, outputs_to_host

CUDA = torch.cuda.is_available()


def _trainer(device: str = 'cpu') -> nnUNetTrainer:
    """A trainer that skipped __init__: only what the methods under test read."""
    t = object.__new__(nnUNetTrainer)
    t.device = torch.device(device)
    t.initial_lr = 1e-2
    t.weight_decay = 3e-5
    t.num_epochs = 1000
    t.current_epoch = 0
    t.is_ddp = False
    t.global_rank = 0
    t.logger = MetaLogger(None, False)
    t.output_folder = 'never_written'  # every method that writes there is mocked
    t.save_every = 50
    t._best_ema = None
    t.plot_progress_every_s = 60
    t._last_progress_plot = None
    t.dataloader_train = t.dataloader_val = None
    return t


def _log_one_epoch(t: nnUNetTrainer, epoch: int):
    # what on_epoch_end reads from the logger
    for key, value in (('train_losses', 0.5), ('val_losses', 0.5), ('dice_per_class_or_region', [0.5]),
                       ('mean_fg_dice', 0.5), ('epoch_start_timestamps', 0.), ('epoch_end_timestamps', 1.)):
        t.logger.log(key, value, epoch)


class TestTrainLoss(unittest.TestCase):
    losses = [0.5, 0.25, 1.0, 0.125, 0.3]

    def test_collate_tensors_like_numpy(self):
        # numpy arrays: what train_step returned before
        ref = collate_outputs([{'loss': torch.tensor(i).numpy()} for i in self.losses])['loss']
        new = collate_outputs([{'loss': torch.tensor(i)} for i in self.losses])['loss']
        self.assertEqual(new.dtype, ref.dtype)
        self.assertEqual(new.shape, ref.shape)
        self.assertTrue(np.array_equal(new, ref))

    def test_logged_train_loss_unchanged(self):
        t, ref = _trainer(), _trainer()
        t.on_train_epoch_end([{'loss': torch.tensor(i)} for i in self.losses])
        ref.on_train_epoch_end([{'loss': torch.tensor(i).numpy()} for i in self.losses])
        self.assertEqual(t.logger.get_value('train_losses', step=-1), ref.logger.get_value('train_losses', step=-1))

    def test_train_step_returns_detached_tensor(self):
        t = _trainer()
        torch.manual_seed(0)
        t.network = torch.nn.Conv3d(1, 2, 3, padding=1)
        t.loss = lambda output, target: ((output - target) ** 2).mean()
        t.optimizer, _ = t.configure_optimizers()
        t.grad_scaler = None
        out = t.train_step({'data': torch.randn(2, 1, 8, 8, 8), 'target': torch.randn(2, 2, 8, 8, 8)})
        self.assertIsInstance(out['loss'], torch.Tensor)
        self.assertFalse(out['loss'].requires_grad)


class TestOutputsToHost(unittest.TestCase):
    def test_same_as_numpy_per_step(self):
        # what the steps returned before: each output's .cpu().numpy(), a 0-d array for the loss
        outputs = [{'loss': torch.tensor(0.5 * i), 'tp_hard': torch.arange(3.) * i, 'name': 'x'} for i in range(4)]
        ref = [{k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in o.items()} for o in outputs]
        for o, r in zip(outputs_to_host(outputs), ref):
            self.assertEqual(o.keys(), r.keys())
            for k in r:
                self.assertEqual(type(o[k]), type(r[k]), k)
                self.assertTrue(np.array_equal(o[k], r[k]), k)
                if isinstance(r[k], np.ndarray):
                    self.assertEqual((o[k].dtype, o[k].shape), (r[k].dtype, r[k].shape), k)

    def test_numpy_and_empty_unchanged(self):
        outputs = [{'loss': np.zeros(())}]
        self.assertIs(outputs_to_host(outputs), outputs)
        self.assertEqual(outputs_to_host([]), [])


class _EpochEndReadsOutputs(nnUNetTrainer):
    """An override that reads the step outputs directly, as custom trainers written for numpy outputs do."""
    def on_train_epoch_end(self, train_outputs):
        self.seen_types = {type(o['loss']) for o in train_outputs}
        self.seen_train = np.mean([o['loss'] for o in train_outputs])

    def on_validation_epoch_end(self, val_outputs):
        self.seen_types |= {type(o[k]) for o in val_outputs for k in o}
        self.seen_val = np.sum([o['tp_hard'] for o in val_outputs], 0)


class TestRunTraining(unittest.TestCase):
    def test_epoch_end_hooks_get_numpy(self):
        t = object.__new__(_EpochEndReadsOutputs)
        t.current_epoch, t.num_epochs = 0, 1
        t.num_iterations_per_epoch, t.num_val_iterations_per_epoch = 3, 2
        t.dataloader_train = t.dataloader_val = iter(lambda: None, 1)
        t.train_step = lambda batch: {'loss': torch.tensor(0.25)}
        t.validation_step = lambda batch: {'loss': torch.tensor(1.), 'tp_hard': torch.tensor([1., 2.])}
        for hook in ('on_train_start', 'on_epoch_start', 'on_train_epoch_start', 'on_validation_epoch_start',
                     'on_epoch_end', 'on_train_end'):
            setattr(t, hook, lambda: None)
        t.run_training()
        # numpy, not tensors: np.mean over CUDA tensors raises
        self.assertEqual(t.seen_types, {np.ndarray})
        self.assertEqual(t.seen_train, 0.25)
        self.assertTrue(np.array_equal(t.seen_val, [2., 4.]))


class TestValidationOutputs(unittest.TestCase):
    def _val_outputs(self, n: int = 3):
        t = _trainer()
        torch.manual_seed(0)
        t.network = torch.nn.Conv3d(1, 3, 3, padding=1)
        t.loss = lambda output, target: output.float().mean()
        t.enable_deep_supervision = False
        t.label_manager = SimpleNamespace(has_regions=False, has_ignore_label=False)
        with torch.no_grad():
            return t, [t.validation_step({'data': torch.randn(2, 1, 8, 8, 8),
                                          'target': torch.randint(0, 3, (2, 1, 8, 8, 8))}) for _ in range(n)]

    def test_validation_step_returns_device_tensors(self):
        _, outputs = self._val_outputs(1)
        for k in ('loss', 'tp_hard', 'fp_hard', 'fn_hard'):
            self.assertIsInstance(outputs[0][k], torch.Tensor, k)
        self.assertEqual(outputs[0]['tp_hard'].shape, (2,))  # background dropped

    def test_logged_values_unchanged(self):
        # the same outputs as numpy arrays, which validation_step returned before
        t, outputs = self._val_outputs()
        ref = _trainer()
        t.on_validation_epoch_end(outputs)
        ref.on_validation_epoch_end([{k: v.cpu().numpy() for k, v in o.items()} for o in outputs])
        for key in ('val_losses', 'mean_fg_dice', 'dice_per_class_or_region'):
            self.assertEqual(t.logger.get_value(key, step=-1), ref.logger.get_value(key, step=-1), key)


class TestExitOnSignal(unittest.TestCase):
    def test_data_loaders_stop_before_the_exit(self):
        t = _trainer()
        t.current_epoch = 10
        _log_one_epoch(t, t.current_epoch)
        t.logger.plot_progress_png = lambda folder: None
        t.dataloader_train = mock.MagicMock(spec=NonDetMultiThreadedAugmenter)
        t.dataloader_val = mock.MagicMock(spec=NonDetMultiThreadedAugmenter)
        with mock.patch.object(nnUNetTrainer, 'save_checkpoint'), \
                mock.patch.object(nnUNetTrainer, 'print_to_log_file'), \
                mock.patch.object(nnUNetTrainer, '_exit_signal_received', lambda self: True):
            with self.assertRaises(SystemExit):
                t.on_epoch_end()
        t.dataloader_train._finish.assert_called_once()
        t.dataloader_val._finish.assert_called_once()


class TestProgressPlot(unittest.TestCase):
    def _plotted(self, epochs, seconds_per_epoch: float, every_s: float = 60):
        t = _trainer()
        t.plot_progress_every_s = every_s
        plotted = []
        t.logger.plot_progress_png = lambda folder: plotted.append(t.current_epoch)
        for e in epochs:
            t.current_epoch = e
            with mock.patch.object(trainer_module, 'time', lambda: 1000. + seconds_per_epoch * e):
                t._maybe_plot_progress(force=e == t.num_epochs - 1)
        return plotted

    def test_throttle(self):
        # 10 s per epoch: the first epoch, then every 6th, then the forced last one
        self.assertEqual(self._plotted(list(range(20)) + [999], 10), [0, 6, 12, 18, 999])

    def test_every_epoch_when_zero_or_slow(self):
        self.assertEqual(self._plotted(range(10), 10, every_s=0), list(range(10)))
        self.assertEqual(self._plotted(range(10), 100), list(range(10)))

    def _epoch_end(self, epoch: int, exit_now: bool, last_plot: float):
        # on_epoch_end of an epoch 5 s after the last plot; returns whether it plotted
        t = _trainer()
        t.current_epoch = epoch
        _log_one_epoch(t, epoch)
        t._last_progress_plot = last_plot
        plotted = []
        t.logger.plot_progress_png = lambda folder: plotted.append(epoch)
        with mock.patch.object(nnUNetTrainer, 'save_checkpoint'), \
                mock.patch.object(nnUNetTrainer, 'print_to_log_file'), \
                mock.patch.object(nnUNetTrainer, '_exit_signal_received', lambda self: exit_now), \
                mock.patch.object(trainer_module, 'time', lambda: last_plot + 5):
            try:
                t.on_epoch_end()
            except SystemExit:
                self.assertTrue(exit_now)
        return bool(plotted)

    def test_epoch_end_forces_last_epoch_and_exit(self):
        self.assertFalse(self._epoch_end(10, exit_now=False, last_plot=2000.))
        self.assertTrue(self._epoch_end(999, exit_now=False, last_plot=2000.))
        self.assertTrue(self._epoch_end(10, exit_now=True, last_plot=2000.))


def _scaled_step(network, optimizer, scaler, grads, inf: bool):
    """One GradScaler step on given gradients, as train_step does after backward; inf puts an overflow into them."""
    scale = scaler.scale(torch.ones((), device='cuda'))
    for p, g in zip(network.parameters(), grads):
        p.grad = None if g is None else g * scale
    if inf:
        next(network.parameters()).grad.view(-1)[0] = float('inf')
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
    scaler.step(optimizer)
    scaler.update()


def _net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.LeakyReLU(), torch.nn.Linear(64, 8)).cuda()


def _foreach_sgd(network):
    # what configure_optimizers built before SGD was fused (foreach is the default on CUDA)
    return torch.optim.SGD(network.parameters(), 1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True)


class TestOptimizer(unittest.TestCase):
    def test_cpu_keeps_foreach_sgd(self):
        t = _trainer('cpu')
        t.network = torch.nn.Linear(4, 4)
        optimizer, _ = t.configure_optimizers()
        self.assertFalse(optimizer.param_groups[0].get('fused'))
        self.assertEqual(len(optimizer.state), 0)


@unittest.skipUnless(CUDA, 'needs CUDA')
class TestFusedSGD(unittest.TestCase):
    def setUp(self):
        net = _net()
        torch.manual_seed(1)
        self.grads = [[torch.randn_like(p) * 1e-2 for p in net.parameters()] for _ in range(4)]

    def _train(self, fused: bool, infs, net=None):
        network = net if net is not None else _net()
        if fused:
            t = _trainer('cuda')
            t.network = network
            optimizer, _ = t.configure_optimizers()
        else:
            optimizer = _foreach_sgd(network)
        scaler = torch.amp.GradScaler('cuda')
        for g, inf in zip(self.grads, infs):
            _scaled_step(network, optimizer, scaler, g, inf)
        return network, optimizer, scaler

    def test_fused_equals_foreach(self):
        # an overflow in the very first step is the case the zero momentum buffers are for
        for infs in ([False] * 4, [True, False, False, False], [False, True, False, False], [True, True, False, False]):
            net_fused, opt_fused, _ = self._train(True, infs)
            self.assertTrue(opt_fused.param_groups[0]['fused'])
            net_foreach, _, _ = self._train(False, infs)
            for a, b in zip(net_fused.parameters(), net_foreach.parameters()):
                self.assertTrue(torch.equal(a, b), f'differs with overflows {infs}')

    def test_continue_foreach_checkpoint_without_some_momentum_buffers(self):
        # a parameter that has not received a gradient yet (e.g. a frozen layer) has no momentum buffer in a foreach
        # checkpoint; fused SGD refuses a mix of present and missing buffers unless _load_optimizer_state fills them
        frozen = 0
        first = [[None if i == frozen else g for i, g in enumerate(step)] for step in self.grads[:2]]
        net_ref = _net()
        opt_ref = _foreach_sgd(net_ref)
        scaler_ref = torch.amp.GradScaler('cuda')
        for g in first:
            _scaled_step(net_ref, opt_ref, scaler_ref, g, False)
        self.assertNotIn('momentum_buffer', opt_ref.state[list(net_ref.parameters())[frozen]])
        ckpt = copy.deepcopy({'net': net_ref.state_dict(), 'opt': opt_ref.state_dict(), 'scaler': scaler_ref.state_dict()})
        for g in self.grads[2:]:
            _scaled_step(net_ref, opt_ref, scaler_ref, g, False)

        t = _trainer('cuda')
        t.network = _net()
        t.network.load_state_dict(ckpt['net'])
        t.optimizer, _ = t.configure_optimizers()
        t._load_optimizer_state(ckpt['opt'])
        scaler = torch.amp.GradScaler('cuda')
        scaler.load_state_dict(ckpt['scaler'])
        for g in self.grads[2:]:
            _scaled_step(t.network, t.optimizer, scaler, g, False)
        for a, b in zip(t.network.parameters(), net_ref.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_continue_foreach_checkpoint_with_fused_sgd(self):
        # 2 foreach steps, checkpoint, 2 more: the reference. A fused trainer that loads the checkpoint must keep fused
        # SGD (without _load_optimizer_state: AssertionError in GradScaler.step) and reach the same parameters
        self.grads, rest = self.grads[:2], self.grads[2:]
        net_ref, opt_ref, scaler_ref = self._train(False, [False, False])
        ckpt = copy.deepcopy({'net': net_ref.state_dict(), 'opt': opt_ref.state_dict(), 'scaler': scaler_ref.state_dict()})
        for g in rest:
            _scaled_step(net_ref, opt_ref, scaler_ref, g, False)

        t = _trainer('cuda')
        t.network = _net()
        t.network.load_state_dict(ckpt['net'])
        t.optimizer, _ = t.configure_optimizers()
        t._load_optimizer_state(ckpt['opt'])
        scaler = torch.amp.GradScaler('cuda')
        scaler.load_state_dict(ckpt['scaler'])
        self.assertTrue(t.optimizer.param_groups[0]['fused'])
        for g in rest:
            _scaled_step(t.network, t.optimizer, scaler, g, False)
        for a, b in zip(t.network.parameters(), net_ref.parameters()):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
