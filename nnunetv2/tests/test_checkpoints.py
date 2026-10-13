import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

import nnunetv2
from nnunetv2.run.run_training import TrainingRunOptions, execute_training
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer
from nnunetv2.training.nnUNetTrainer.variants.lr_schedule.nnUNetTrainer_warmup import nnUNetTrainer_warmup
from nnunetv2.utilities import checkpoints, convert_checkpoints
from nnunetv2.utilities.checkpoints import assert_checkpoints_equal, find_checkpoint_file, load_checkpoint_file, \
    load_network_weights, read_checkpoint_metadata, resolve_checkpoint_file, save_checkpoint_file, \
    without_training_state


def _unet_kwargs(norm):
    return dict(input_channels=2, n_stages=4, features_per_stage=[8, 16, 32, 32], conv_op=nn.Conv3d,
                kernel_sizes=3, strides=[1, 2, 2, 2], num_classes=3, conv_bias=True, norm_op=norm,
                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, deep_supervision=False)


def _build_network(name: str):
    torch.manual_seed(0)
    if name == 'PlainConvUNet':
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        net = PlainConvUNet(n_conv_per_stage=2, n_conv_per_stage_decoder=2, **_unet_kwargs(nn.BatchNorm3d))
        net.train()
        net(torch.randn(2, 2, 32, 32, 32))  # running stats and num_batches_tracked (int64) get values
        return net, (1, 2, 32, 32, 32)
    if name == 'ResidualEncoderUNet':
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
        return ResidualEncoderUNet(n_blocks_per_stage=[1, 2, 2, 2], n_conv_per_stage_decoder=1,
                                   **_unet_kwargs(nn.InstanceNorm3d)), (1, 2, 32, 32, 32)
    if name == 'Primus':
        from dynamic_network_architectures.architectures.primus import Primus
        return Primus(input_channels=1, embed_dim=96, patch_embed_size=(8, 8, 8), num_classes=3, eva_depth=2,
                      eva_numheads=4, input_shape=(32, 32, 32)), (1, 1, 32, 32, 32)
    raise ValueError(name)


def _training_checkpoint(optimizer_class=torch.optim.SGD, **optimizer_kwargs) -> dict:
    """A checkpoint shaped like the one nnUNetTrainer.save_checkpoint writes, after a few training steps."""
    if not optimizer_kwargs:
        optimizer_kwargs = dict(lr=0.01, momentum=0.99, nesterov=True)  # what nnUNetTrainer uses
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv3d(1, 4, 3), nn.InstanceNorm3d(4, affine=True), nn.LeakyReLU(), nn.Conv3d(4, 2, 1))
    opt = optimizer_class(net.parameters(), **optimizer_kwargs)
    for _ in range(2):
        opt.zero_grad()
        net(torch.randn(2, 1, 8, 8, 8)).square().mean().backward()
        opt.step()
    return {
        'network_weights': net.state_dict(),
        'optimizer_state': opt.state_dict(),
        'grad_scaler_state': {'scale': 65536.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000,
                              '_growth_tracker': 17},
        'logging': {
            'mean_fg_dice': [np.float64(0.5), np.float64(np.nan)],
            'ema_fg_dice': [np.float64(0.5), np.float64(0.55)],
            'dice_per_class_or_region': [[np.float64(0.4), np.float64(0.6)], [np.float64(np.nan), 1.0]],
            'train_losses': [np.float32(-0.25), np.float32(-0.5)],
            'val_losses': [np.float32(-0.2), np.float32(-0.4)],
            'lrs': [0.01, 0.0099],
            'epoch_start_timestamps': [1760000000.123456, 1760000100.654321],
            'epoch_end_timestamps': [1760000099.5, 1760000199.25],
        },
        '_best_ema': np.float64(0.55),
        'current_epoch': 2,
        'init_args': {'plans': {'plans_name': 'nnUNetPlans', 'configurations': {'3d_fullres': {
            'patch_size': [8, 8, 8], 'architecture': {
                'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
                '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}}}},
            'configuration': '3d_fullres', 'fold': 0, 'dataset_json': {'labels': {'background': 0, 'a': 1}},
            'device': torch.device('cuda', 0)},
        'trainer_name': 'nnUNetTrainer',
        'inference_allowed_mirroring_axes': (0, 1, 2),
    }


class _TmpDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name: str) -> str:
        return os.path.join(self.tmp, name)

    def roundtrip(self, checkpoint: dict, **load_kwargs) -> dict:
        f = self.path('roundtrip.safetensors')
        save_checkpoint_file(checkpoint, f)
        return load_checkpoint_file(f, **load_kwargs)


class TestArchitectures(_TmpDirTestCase):
    def test_roundtrip_strict_load_identical_output(self):
        for name in ('PlainConvUNet', 'ResidualEncoderUNet', 'Primus'):
            with self.subTest(name):
                net, shape = _build_network(name)
                net.eval()
                sd = net.state_dict()
                loaded = self.roundtrip({'network_weights': sd})['network_weights']
                assert_checkpoints_equal(sd, loaded)

                fresh, _ = _build_network(name)
                for p in fresh.parameters():  # make sure everything really comes from the file
                    p.data.zero_()
                fresh.load_state_dict(loaded, strict=True)
                fresh.eval()
                x = torch.randn(shape)
                with torch.no_grad():
                    self.assertTrue(torch.equal(net(x), fresh(x)))

    def test_shared_tensors_are_stored_once(self):
        net, _ = _build_network('ResidualEncoderUNet')
        sd = net.state_dict()
        storages = {}
        for t in sd.values():
            storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        unique_bytes = sum(storages.values())
        self.assertLess(len(storages), len(sd))  # the test is meaningless without aliases
        f = self.path('w.safetensors')
        save_checkpoint_file({'network_weights': sd}, f)
        self.assertLess(os.path.getsize(f), 1.1 * unique_bytes)
        # stored under their own names, readable by the safetensors library itself
        stored = load_file(f)
        self.assertTrue(all(k in sd for k in stored if k != '__nnunet_checkpoint_state__'))

    def test_partly_overlapping_tensors(self):
        class PartialOverlap(nn.Module):
            def __init__(self):
                super().__init__()
                self.flat = nn.Parameter(torch.randn(10))
                self.register_buffer('head', self.flat.data[2:6])  # same storage, different view

        sd = PartialOverlap().state_dict()
        loaded = self.roundtrip({'network_weights': sd})['network_weights']
        assert_checkpoints_equal(sd, loaded)
        fresh = PartialOverlap()
        fresh.load_state_dict(loaded)
        self.assertTrue(torch.equal(fresh.head, sd['head']))

    def test_state_dict_metadata_is_kept(self):
        # load_state_dict hands each module its version from _metadata; torch.save keeps it, so must we
        net, _ = _build_network('PlainConvUNet')
        sd = net.state_dict()
        loaded = self.roundtrip({'network_weights': sd})['network_weights']
        self.assertEqual(loaded._metadata, sd._metadata)
        with self.assertRaises(AssertionError):
            assert_checkpoints_equal(sd, OrderedDict(sd))  # same items, no _metadata

    def test_extra_state_is_refused(self):
        class ExtraState(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

            def get_extra_state(self):
                return {'calibration': object()}

            def set_extra_state(self, state):
                pass

        f = self.path('x.safetensors')
        with self.assertRaises(ValueError) as e:
            save_checkpoint_file({'network_weights': ExtraState().state_dict()}, f)
        self.assertIn('_extra_state', str(e.exception))
        self.assertFalse(os.path.exists(f))
        self.assertFalse(os.path.exists(f + '.tmp'))


class TestTrainingCheckpoint(_TmpDirTestCase):
    def test_roundtrip_keeps_values_and_types(self):
        for optimizer_class, kwargs in ((torch.optim.SGD, dict(lr=0.01, momentum=0.99, nesterov=True)),
                                        (torch.optim.AdamW, dict(lr=1e-3, betas=(0.9, 0.98)))):
            with self.subTest(optimizer_class.__name__):
                ckpt = _training_checkpoint(optimizer_class, **kwargs)
                loaded = self.roundtrip(ckpt)
                assert_checkpoints_equal(ckpt, loaded)
                self.assertIs(type(loaded['logging']['train_losses'][0]), np.float32)
                self.assertIs(type(loaded['_best_ema']), np.float64)
                self.assertEqual(loaded['inference_allowed_mirroring_axes'], (0, 1, 2))
                self.assertIsInstance(loaded['network_weights'], OrderedDict)
                self.assertEqual(list(loaded['optimizer_state']['state'].keys()), [0, 1, 2, 3, 4, 5])
                self.assertEqual(loaded['init_args']['device'], torch.device('cuda', 0))

    def test_mirroring_axes_none(self):
        ckpt = _training_checkpoint()
        ckpt['inference_allowed_mirroring_axes'] = None
        self.assertIsNone(self.roundtrip(ckpt)['inference_allowed_mirroring_axes'])

    def test_resumed_optimizer_takes_identical_steps(self):
        for optimizer_class, kwargs in ((torch.optim.SGD, dict(lr=0.01, momentum=0.99, nesterov=True)),
                                        (torch.optim.AdamW, dict(lr=1e-3))):
            with self.subTest(optimizer_class.__name__):
                ckpt = _training_checkpoint(optimizer_class, **kwargs)
                pth = self.path('c.pth')
                torch.save(ckpt, pth)
                st = self.path('c.safetensors')
                save_checkpoint_file(ckpt, st)

                params_after = []
                for f in (pth, st):
                    loaded = load_checkpoint_file(f)
                    net = nn.Sequential(nn.Conv3d(1, 4, 3), nn.InstanceNorm3d(4, affine=True), nn.LeakyReLU(),
                                        nn.Conv3d(4, 2, 1))
                    net.load_state_dict(loaded['network_weights'])
                    opt = optimizer_class(net.parameters(), **kwargs)
                    opt.load_state_dict(loaded['optimizer_state'])
                    torch.manual_seed(1)
                    for _ in range(3):
                        opt.zero_grad()
                        net(torch.randn(2, 1, 8, 8, 8)).square().mean().backward()
                        opt.step()
                    params_after.append(net.state_dict())
                assert_checkpoints_equal(*params_after)

    def test_without_training_state(self):
        ckpt = _training_checkpoint()
        final = without_training_state(ckpt)
        self.assertNotIn('optimizer_state', final)
        self.assertNotIn('grad_scaler_state', final)
        loaded = self.roundtrip(final)
        assert_checkpoints_equal(final, loaded)
        self.assertFalse(read_checkpoint_metadata(self.path('roundtrip.safetensors'))['contains_training_state'])

    def test_header_metadata(self):
        save_checkpoint_file(_training_checkpoint(), self.path('c.safetensors'))
        meta = read_checkpoint_metadata(self.path('c.safetensors'))
        self.assertEqual(meta['format'], 'nnunetv2-checkpoint')
        self.assertEqual(meta['trainer_name'], 'nnUNetTrainer')
        self.assertEqual(meta['configuration'], '3d_fullres')
        self.assertEqual(meta['fold'], 0)
        self.assertEqual(meta['current_epoch'], 2)
        self.assertEqual(meta['inference_allowed_mirroring_axes'], [0, 1, 2])
        self.assertTrue(meta['contains_training_state'])

    def test_load_selected_keys(self):
        ckpt = _training_checkpoint()
        loaded = self.roundtrip(ckpt, keys=('network_weights', 'trainer_name', 'not_there'))
        self.assertEqual(list(loaded.keys()), ['network_weights', 'trainer_name'])
        assert_checkpoints_equal(ckpt['network_weights'], loaded['network_weights'])

    def test_map_location(self):
        loaded = self.roundtrip(_training_checkpoint(), map_location=torch.device('cpu'))
        self.assertEqual(loaded['network_weights']['0.weight'].device, torch.device('cpu'))
        if torch.cuda.is_available():
            loaded = self.roundtrip(_training_checkpoint(), map_location=torch.device('cuda', 0))
            self.assertEqual(loaded['optimizer_state']['state'][0]['momentum_buffer'].device.type, 'cuda')

    def test_plain_data_in_odd_places(self):
        ckpt = {'network_weights': OrderedDict(w=torch.arange(3.)),
                'extra': {(1, 'a'): [np.int64(3), np.bool_(True), np.float16(0.1)], '$type': 'not a tag',
                          'arr': np.arange(6, dtype=np.int16).reshape(2, 3), 'dtype': torch.bfloat16,
                          'scalar': torch.tensor(1.5), 'bf16': torch.ones(2, dtype=torch.bfloat16),
                          'inf': float('inf'), 'empty': torch.zeros(0), 'nested': ((), [], {}),
                          'zero_d': np.array(0.5)}}
        loaded = self.roundtrip(ckpt)
        assert_checkpoints_equal(ckpt, loaded)
        self.assertEqual(loaded['extra']['zero_d'].shape, ())

    def test_conjugate_and_negative_views(self):
        # same memory and strides as the tensor they view, but other values: must not be stored as aliases of it
        x = torch.randn(4, dtype=torch.complex64)
        ckpt = {'network_weights': OrderedDict(x=x, conj=x.conj(), neg_imag=x.conj().imag, imag=x.imag)}
        self.assertTrue(ckpt['network_weights']['neg_imag'].is_neg())
        loaded = self.roundtrip(ckpt)
        assert_checkpoints_equal(ckpt, loaded)
        self.assertTrue(torch.equal(loaded['network_weights']['conj'], x.conj()))
        self.assertTrue(torch.equal(loaded['network_weights']['neg_imag'], -x.imag))

    def test_reserved_tensor_names(self):
        ckpt = {'network_weights': OrderedDict([('__metadata__', torch.ones(2)),
                                                ('__nnunet_checkpoint_state__', torch.zeros(3))])}
        assert_checkpoints_equal(ckpt, self.roundtrip(ckpt))

    def test_unsupported_entries_are_named_and_nothing_is_written(self):
        ckpt = _training_checkpoint()
        ckpt['init_args']['callback'] = lambda x: x
        ckpt['logging']['sparse'] = torch.eye(2).to_sparse()
        ckpt['logging']['big_endian'] = np.arange(3, dtype='>f4')
        f = self.path('c.safetensors')
        with self.assertRaises(ValueError) as e:
            save_checkpoint_file(ckpt, f)
        self.assertIn('init_args/callback', str(e.exception))
        self.assertIn('logging/sparse', str(e.exception))
        self.assertIn('logging/big_endian', str(e.exception))
        self.assertFalse(os.path.exists(f))

    def test_failed_write_keeps_previous_file(self):
        f = self.path('c.safetensors')
        first = _training_checkpoint()
        save_checkpoint_file(first, f)

        def write_half_and_die(tensors, metadata, filename):
            with open(filename, 'wb') as fh:
                fh.write(b'12345')
            raise KeyboardInterrupt

        with mock.patch.object(checkpoints, '_write_safetensors', write_half_and_die):
            with self.assertRaises(KeyboardInterrupt):
                save_checkpoint_file({'network_weights': {}}, f)
        self.assertFalse(os.path.exists(f + '.tmp'))
        assert_checkpoints_equal(first, load_checkpoint_file(f))

    def test_loaded_tensors_do_not_read_from_the_file(self):
        # safetensors maps the file; the loaded tensors must not see the file change underneath them
        f, plain = self.path('c.safetensors'), self.path('plain.safetensors')
        save_checkpoint_file({'network_weights': OrderedDict(w=torch.ones(1000))}, f)
        save_file({'w': torch.ones(1000)}, plain)
        loaded = [load_checkpoint_file(f)['network_weights']['w'], load_network_weights(f)['w'],
                  load_network_weights(plain)['w']]
        for fn in (f, plain):
            with open(fn, 'r+b') as fh:  # in place, same inode, like cp onto an existing file
                header_length = int.from_bytes(fh.read(8), 'little')
                fh.seek(8 + header_length)
                fh.write(bytes(os.path.getsize(fn) - 8 - header_length))
        for t in loaded:
            self.assertTrue(torch.equal(t, torch.ones(1000)))

    def test_comparison_survives_python_O(self):
        # the converter deletes the .pth once assert_checkpoints_equal passes, so it must not consist of asserts
        code = ('import numpy as np, torch\n'
                'from nnunetv2.utilities.checkpoints import assert_checkpoints_equal\n'
                'pairs = [(torch.ones(2), torch.zeros(2)), (torch.ones(2), torch.ones(3)), (1, 2), ({"a": 1}, {"b": 1}),\n'
                '         ((1,), [1]), ([1], [1, 2]), (np.arange(2), np.arange(2) + 1), (float("nan"), 1.0)]\n'
                'for a, b in pairs:\n'
                '    try:\n'
                '        assert_checkpoints_equal(a, b)\n'
                '    except AssertionError:\n'
                '        continue\n'
                '    raise SystemExit(f"no difference found between {a!r} and {b!r}")\n')
        repo = os.path.dirname(os.path.dirname(nnunetv2.__file__))
        env = dict(os.environ, PYTHONPATH=repo + os.pathsep + os.environ.get('PYTHONPATH', ''))
        result = subprocess.run([sys.executable, '-O', '-c', code], cwd=self.tmp, env=env, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


class TestLegacyAndForeignFiles(_TmpDirTestCase):
    def test_legacy_pth_loads_like_torch_load(self):
        ckpt = _training_checkpoint()
        variants = {'current': ckpt, 'old, no mirroring axes': {k: v for k, v in ckpt.items()
                                                                  if k != 'inference_allowed_mirroring_axes'}}
        for name, c in variants.items():
            with self.subTest(name):
                f = self.path('checkpoint_final.pth')
                torch.save(c, f)
                reference = torch.load(f, map_location='cpu', weights_only=False)
                assert_checkpoints_equal(reference, load_checkpoint_file(f, map_location='cpu'))
                # .pth files are always loaded in full
                assert_checkpoints_equal(reference, load_checkpoint_file(f, keys=('network_weights',)))
                assert_checkpoints_equal(reference['network_weights'], load_network_weights(f))

    def test_pickle_with_safetensors_name_is_not_loaded(self):
        f = self.path('checkpoint_final.safetensors')
        torch.save(_training_checkpoint(), f)
        with self.assertRaises(ValueError) as e:
            load_checkpoint_file(f)
        self.assertIn('rename it to .pth', str(e.exception))

    def test_plain_safetensors_weights(self):
        net, _ = _build_network('PlainConvUNet')
        sd = {k: v.clone() for k, v in net.state_dict().items()}
        f = self.path('weights.safetensors')
        save_file(sd, f)
        loaded = load_network_weights(f)
        assert_checkpoints_equal(sd, {k: loaded[k] for k in sd})  # the safetensors library sorts the names
        with self.assertRaises(ValueError):
            load_checkpoint_file(f)
        # header metadata of other writers is not JSON (transformers and timm write {"format": "pt"})
        save_file(sd, f, metadata={'format': 'pt'})
        loaded = load_network_weights(f)
        assert_checkpoints_equal(sd, {k: loaded[k] for k in sd})
        with self.assertRaises(ValueError) as e:
            load_checkpoint_file(f)
        self.assertIn('not an nnU-Net checkpoint', str(e.exception))

    def test_newer_format_version_is_refused(self):
        f = self.path('c.safetensors')
        with mock.patch.object(checkpoints, 'FORMAT_VERSION', checkpoints.FORMAT_VERSION + 1):
            save_checkpoint_file({'network_weights': {}}, f)
        with self.assertRaises(ValueError) as e:
            load_checkpoint_file(f)
        self.assertIn('newer nnU-Net', str(e.exception))


class TestResolve(_TmpDirTestCase):
    def touch(self, name):
        open(self.path(name), 'w').close()
        return self.path(name)

    def test_named_file_wins(self):
        pth, st = self.touch('checkpoint_final.pth'), self.touch('checkpoint_final.safetensors')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(resolve_checkpoint_file(pth), pth)
            self.assertEqual(resolve_checkpoint_file(st), st)
        self.assertEqual(len(w), 2)

    def test_fallback_to_other_format(self):
        pth = self.touch('checkpoint_best.pth')
        self.assertEqual(resolve_checkpoint_file(self.path('checkpoint_best.safetensors')), pth)
        st = self.touch('checkpoint_final.safetensors')
        self.assertEqual(resolve_checkpoint_file(self.path('checkpoint_final.pth')), st)

    def test_missing(self):
        with self.assertRaises(FileNotFoundError) as e:
            resolve_checkpoint_file(self.path('checkpoint_final.safetensors'))
        self.assertIn('checkpoint_final.safetensors', str(e.exception))
        self.assertIn('checkpoint_final.pth', str(e.exception))
        self.assertIsNone(find_checkpoint_file(self.path('checkpoint_final.pth')))

    def test_loading_resolves_names(self):
        ckpt = _training_checkpoint()
        save_checkpoint_file(ckpt, self.path('checkpoint_final.safetensors'))
        torch.save(ckpt, self.path('checkpoint_best.pth'))
        for name in ('checkpoint_final.pth', 'checkpoint_best.safetensors'):
            with self.subTest(name):
                assert_checkpoints_equal(ckpt, load_checkpoint_file(self.path(name)))
                assert_checkpoints_equal(ckpt['network_weights'], load_network_weights(self.path(name)))

    def test_other_extensions_have_no_fallback(self):
        self.touch('model.pth')
        self.assertIsNone(find_checkpoint_file(self.path('model.pt')))


class _Unpicklable:
    """Picklable, but not something safetensors + JSON can hold."""
    pass


class _FakeLogger:
    def __init__(self, state=None):
        self.state = state

    def get_checkpoint(self):
        return self.state

    def load_checkpoint(self, state):
        self.state = state


def _fake_trainer(ckpt: dict, num_epochs: int = 10) -> SimpleNamespace:
    """
    Just what nnUNetTrainer.save_checkpoint and load_checkpoint touch, so that the real methods can be called on it
    without plans, data or a GPU.
    """
    net = nn.Sequential(nn.Conv3d(1, 4, 3), nn.InstanceNorm3d(4, affine=True), nn.LeakyReLU(), nn.Conv3d(4, 2, 1))
    net.load_state_dict(ckpt['network_weights'])
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    opt.load_state_dict(ckpt['optimizer_state'])
    messages = []
    return SimpleNamespace(
        global_rank=0, disable_checkpointing=False, is_ddp=False, was_initialized=True, device=torch.device('cpu'),
        network=net, optimizer=opt, grad_scaler=None, logger=_FakeLogger(ckpt['logging']), _best_ema=ckpt['_best_ema'],
        current_epoch=ckpt['current_epoch'] - 1,  # save_checkpoint stores current_epoch + 1
        num_epochs=num_epochs, my_init_kwargs=ckpt['init_args'],
        inference_allowed_mirroring_axes=ckpt['inference_allowed_mirroring_axes'],
        print_to_log_file=lambda *args, **kwargs: messages.append(' '.join(str(a) for a in args)), messages=messages)


class TestTrainerCheckpoints(_TmpDirTestCase):
    def test_save_writes_safetensors_and_replaces_legacy_file(self):
        ckpt = _training_checkpoint()
        trainer = _fake_trainer(ckpt)
        for name in ('checkpoint_best', 'checkpoint_latest', 'checkpoint_final'):
            with self.subTest(name):
                legacy = self.path(f'{name}.pth')
                torch.save({'stale': True}, legacy)
                nnUNetTrainer.save_checkpoint(trainer, legacy)  # a .pth name is written as .safetensors
                self.assertFalse(os.path.exists(legacy))
                loaded = load_checkpoint_file(self.path(f'{name}.safetensors'))
                # the fake trainer has no grad scaler
                expected = dict(ckpt, trainer_name='SimpleNamespace', grad_scaler_state=None)
                if name == 'checkpoint_final':
                    expected = without_training_state(expected)
                assert_checkpoints_equal(expected, loaded)

    def test_unstorable_checkpoint_raises_on_every_rank(self):
        # only rank 0 writes, but a rank that raised alone would leave the others waiting in a collective under DDP
        ckpt = _training_checkpoint()
        for rank in (0, 1):
            with self.subTest(rank=rank):
                trainer = _fake_trainer(ckpt)
                trainer.global_rank = rank
                nnUNetTrainer.save_checkpoint(trainer, self.path(f'good_{rank}.safetensors'))
                self.assertEqual(os.path.isfile(self.path(f'good_{rank}.safetensors')), rank == 0)
                trainer.my_init_kwargs = dict(ckpt['init_args'], something=_Unpicklable())
                with self.assertRaisesRegex(ValueError, 'init_args/something'):
                    nnUNetTrainer.save_checkpoint(trainer, self.path(f'bad_{rank}.safetensors'))
                self.assertFalse(os.path.exists(self.path(f'bad_{rank}.safetensors')))

    def test_load_legacy_and_safetensors_give_the_same_trainer_state(self):
        ckpt = _training_checkpoint()
        torch.save(ckpt, self.path('checkpoint_latest.pth'))
        save_checkpoint_file(ckpt, self.path('checkpoint_best.safetensors'))
        states = []
        # named as written, and named in the other format (resolved to the file that exists)
        for name in ('checkpoint_latest.pth', 'checkpoint_best.safetensors', 'checkpoint_latest.safetensors',
                     'checkpoint_best.pth'):
            trainer = _fake_trainer(_training_checkpoint(torch.optim.SGD, lr=0.5, momentum=0.5))  # different state
            nnUNetTrainer.load_checkpoint(trainer, self.path(name))
            states.append({'network_weights': trainer.network.state_dict(),
                           'optimizer_state': trainer.optimizer.state_dict(), 'logging': trainer.logger.state,
                           '_best_ema': trainer._best_ema, 'current_epoch': trainer.current_epoch,
                           'init_args': trainer.my_init_kwargs,
                           'inference_allowed_mirroring_axes': trainer.inference_allowed_mirroring_axes})
        for s in states:
            assert_checkpoints_equal(states[0], s)
        assert_checkpoints_equal({k: ckpt[k] for k in states[0]}, states[0])

    def test_load_final_without_optimizer_state(self):
        # nnUNetTrainer_warmup and PretrainedTrainer have their own copy of load_checkpoint, so they are checked too,
        # with a file name and with a checkpoint dict
        ckpt = _training_checkpoint()
        final = self.path('checkpoint_final.safetensors')
        save_checkpoint_file(without_training_state(ckpt), final)
        for trainer_class in (nnUNetTrainer, nnUNetTrainer_warmup, PretrainedTrainer):
            for as_dict in (False, True):
                for num_epochs, warns in ((ckpt['current_epoch'], False), (ckpt['current_epoch'] + 5, True)):
                    with self.subTest(trainer_class.__name__, as_dict=as_dict, num_epochs=num_epochs):
                        trainer = _fake_trainer(ckpt, num_epochs=num_epochs)
                        # what the two subclasses need to set up their optimizer while loading
                        trainer.warmup_duration_whole_net = 0
                        trainer.get_stage = lambda: 'train'
                        trainer.configure_optimizers = lambda stage, t=trainer: (t.optimizer, None)
                        optimizer_before = trainer.optimizer.state_dict()
                        trainer_class.load_checkpoint(trainer, load_checkpoint_file(final) if as_dict else final)
                        assert_checkpoints_equal(ckpt['network_weights'], trainer.network.state_dict())
                        assert_checkpoints_equal(optimizer_before, trainer.optimizer.state_dict())
                        self.assertEqual(any('no optimizer state' in m for m in trainer.messages), warns)


class TestConverter(_TmpDirTestCase):
    def make_training(self, folder: str, names=('checkpoint_final.pth', 'checkpoint_best.pth')) -> dict:
        os.makedirs(folder, exist_ok=True)
        ckpt = _training_checkpoint()
        for n in names:
            torch.save(ckpt, os.path.join(folder, n))
        return ckpt

    def test_convert_final_and_best(self):
        fold = self.path('model/fold_0')
        ckpt = self.make_training(fold)
        for f in convert_checkpoints.find_checkpoints_to_convert([self.path('model')],
                                                                 ('checkpoint_final.pth', 'checkpoint_best.pth')):
            convert_checkpoints.convert_checkpoint(f)
        self.assertEqual(sorted(os.listdir(fold)), ['checkpoint_best.safetensors', 'checkpoint_final.safetensors'])
        assert_checkpoints_equal(without_training_state(ckpt),
                                 load_checkpoint_file(os.path.join(fold, 'checkpoint_final.safetensors')))
        assert_checkpoints_equal(ckpt, load_checkpoint_file(os.path.join(fold, 'checkpoint_best.safetensors')))

    def test_keep_pth_and_overwrite(self):
        fold = self.path('fold_0')
        self.make_training(fold)
        pth = os.path.join(fold, 'checkpoint_best.pth')
        self.assertIsNotNone(convert_checkpoints.convert_checkpoint(pth, keep_pth=True))
        self.assertTrue(os.path.isfile(pth))
        self.assertIsNone(convert_checkpoints.convert_checkpoint(pth))  # exists -> skipped, .pth untouched
        self.assertTrue(os.path.isfile(pth))
        self.assertIsNotNone(convert_checkpoints.convert_checkpoint(pth, overwrite=True))
        self.assertFalse(os.path.isfile(pth))

    def test_scan_does_not_warn_about_kept_pth(self):
        # --keep_pth leaves both formats behind on purpose; scanning such a folder again must not warn about that
        fold = self.path('fold_0')
        self.make_training(fold)
        names = ('checkpoint_final.pth', 'checkpoint_best.pth')
        for f in convert_checkpoints.find_checkpoints_to_convert([fold], names):
            convert_checkpoints.convert_checkpoint(f, keep_pth=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            found = convert_checkpoints.find_checkpoints_to_convert([fold], names)
        self.assertEqual(sorted(os.path.basename(f) for f in found), sorted(names))
        self.assertEqual([str(m.message) for m in w], [])

    def test_unfinished_trainings_are_skipped(self):
        self.make_training(self.path('finished/fold_0'))
        self.make_training(self.path('running/fold_0'), ('checkpoint_best.pth', 'checkpoint_latest.pth'))
        self.make_training(self.path('crashed_late/fold_0'), ('checkpoint_final.pth', 'checkpoint_latest.pth'))
        found = convert_checkpoints.find_checkpoints_to_convert([self.tmp], ('checkpoint_final.pth',
                                                                            'checkpoint_best.pth'))
        self.assertEqual(sorted(os.path.relpath(f, self.tmp) for f in found),
                         ['finished/fold_0/checkpoint_best.pth', 'finished/fold_0/checkpoint_final.pth'])

    def test_failures_keep_the_pth(self):
        fold = self.path('fold_0')
        ckpt = self.make_training(fold, ('checkpoint_best.pth',))
        pth = os.path.join(fold, 'checkpoint_best.pth')

        def corrupted(filename, *args, **kwargs):
            loaded = load_checkpoint_file(filename, *args, **kwargs)
            loaded['current_epoch'] += 1
            return loaded

        with mock.patch.object(convert_checkpoints, 'load_checkpoint_file', corrupted):
            with self.assertRaises(RuntimeError):
                convert_checkpoints.convert_checkpoint(pth)
        self.assertEqual(os.listdir(fold), ['checkpoint_best.pth'])

        def unreadable(filename, *args, **kwargs):
            raise ValueError('cannot read it back')

        with mock.patch.object(convert_checkpoints, 'load_checkpoint_file', unreadable):
            with self.assertRaises(ValueError):
                convert_checkpoints.convert_checkpoint(pth)
        self.assertEqual(os.listdir(fold), ['checkpoint_best.pth'])

        ckpt['init_args']['something'] = _Unpicklable()
        torch.save(ckpt, pth)
        with self.assertRaises(ValueError):
            convert_checkpoints.convert_checkpoint(pth)
        self.assertEqual(os.listdir(fold), ['checkpoint_best.pth'])

    def test_read_back_never_falls_back_to_the_pth(self):
        # if the new file is gone before it is read back (another process), the read-back must not find the .pth of
        # the same name instead: the .pth would be compared with itself and deleted
        fold = self.path('fold_0')
        self.make_training(fold, ('checkpoint_best.pth',))
        pth = os.path.join(fold, 'checkpoint_best.pth')
        save = convert_checkpoints.save_checkpoint_file

        def vanishing(checkpoint, filename):
            save(checkpoint, filename)
            os.remove(filename)

        with mock.patch.object(convert_checkpoints, 'save_checkpoint_file', vanishing):
            with self.assertRaises(FileNotFoundError):
                convert_checkpoints.convert_checkpoint(pth)
        self.assertEqual(os.listdir(fold), ['checkpoint_best.pth'])

    def test_target_holds_only_verified_files(self):
        fold = self.path('fold_0')
        self.make_training(fold, ('checkpoint_best.pth',))
        pth = os.path.join(fold, 'checkpoint_best.pth')
        target = os.path.join(fold, 'checkpoint_best.safetensors')

        def read_back(filename, *args, **kwargs):
            self.assertFalse(os.path.isfile(target), 'the file is under its final name before it is verified')
            return load_checkpoint_file(filename, *args, **kwargs)

        with mock.patch.object(convert_checkpoints, 'load_checkpoint_file', read_back):
            convert_checkpoints.convert_checkpoint(pth, keep_pth=True)
        self.assertEqual(sorted(os.listdir(fold)), ['checkpoint_best.pth', 'checkpoint_best.safetensors'])

        # a failed overwrite leaves the earlier conversion as it was
        def unreadable(filename, *args, **kwargs):
            raise ValueError('cannot read it back')

        with mock.patch.object(convert_checkpoints, 'load_checkpoint_file', unreadable):
            with self.assertRaises(ValueError):
                convert_checkpoints.convert_checkpoint(pth, overwrite=True)
        self.assertEqual(sorted(os.listdir(fold)), ['checkpoint_best.pth', 'checkpoint_best.safetensors'])
        assert_checkpoints_equal(torch.load(pth, weights_only=False), load_checkpoint_file(target))

    def test_missing_input_folder_is_an_error(self):
        with mock.patch('sys.argv', ['nnUNetv2_convert_checkpoints_to_safetensors', '-i', self.path('not_there')]):
            with self.assertRaises(SystemExit) as e:
                convert_checkpoints.convert_checkpoints_entry()
        self.assertNotEqual(e.exception.code, 0)


class TestRunTraining(_TmpDirTestCase):
    def test_val_best_passes_the_file_that_exists(self):
        # custom trainers may override load_checkpoint with a plain torch.load, which does not resolve names
        torch.save({}, self.path('checkpoint_best.pth'))
        loaded = []
        trainer = SimpleNamespace(output_folder=self.tmp, run_training=lambda: None, load_checkpoint=loaded.append,
                                  perform_actual_validation=lambda *args: None)
        execute_training(lambda device: trainer, torch.device('cpu'), TrainingRunOptions(val_with_best=True))
        self.assertEqual(loaded, [self.path('checkpoint_best.pth')])


if __name__ == '__main__':
    unittest.main()
