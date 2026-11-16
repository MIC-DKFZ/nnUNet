"""
Streaming final validation (variants/validation/nnUNetTrainer_streamingValidation.py): the predictor the trainer feeds
with its live network, the resampling notice, and the trainers built on the mixin.
"""
import os
import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import torch

from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from nnunetv2.tests.test_predict_streaming_feature_ring import GEOMETRIES, KERNELS, _Base, _case, _net, _predictor
from nnunetv2.training.loss.chunked_losses import ChunkedMemoryEfficientSoftDiceLoss, DC_and_CE_loss_chunked
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.validation.nnUNetTrainer_streamingValidation import (
    StreamingValidationMixin,
    nnUNetPredictorStreamingFromTrainer,
)
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


class TestTrainerValidationPredictor(_Base):
    """nnUNetPredictorStreamingFromTrainer: heads from the live network, which goes back to the trainer intact."""

    def _check(self, device, compiled):
        # manual_initialization compiles an uncompiled network when nnUNet_compile is set: test both cases explicitly
        with mock.patch.dict(os.environ, {'nnUNet_compile': 'False'}):
            self._check_impl(device, compiled)

    def _check_impl(self, device, compiled):
        trainer_net = _net().to(device)
        trainer_network = torch.compile(trainer_net) if compiled else trainer_net
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        with torch.inference_mode():
            before = trainer_network(x).clone()
        data, props = _case(*GEOMETRIES[0][:2])
        segs = {}
        for fr in (False, True):
            p = nnUNetPredictorStreamingFromTrainer(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                                    device=device, allow_tqdm=False, feature_ring=fr)
            ref = _predictor(device)
            ref.plans_manager.get_label_manager = lambda dataset_json: ref.label_manager
            p.manual_initialization(trainer_network, ref.plans_manager, ref.configuration_manager, [None], {},
                                    'nnUNetTrainer', (0, 1, 2))
            p.setup_feature_ring()
            self.assertEqual(p._detached_head is not None, fr)
            if fr:
                self.assertIsInstance(trainer_net.decoder.seg_layers[-1], torch.nn.Identity)
                self.assertEqual(hasattr(p.network, '_orig_mod'), compiled)
            try:
                segs[fr] = p.predict_preprocessed_case(data, props)
            finally:
                p.restore_head()
            self.assertIsInstance(trainer_net.decoder.seg_layers[-1], torch.nn.Conv3d)
        # the validation's fp16 ring: the two rings round different quantities to fp16
        self.assertSameSegmentation(segs[True], segs[False], 0.998)
        with torch.inference_mode():
            after = trainer_network(x)
        self.assertEqual(after.shape, before.shape)
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_cpu(self):
        self._check(torch.device('cpu'), compiled=False)

    @unittest.skipUnless(KERNELS, 'CUDA + Triton')
    def test_cuda_compiled(self):
        self._check(torch.device('cuda'), compiled=True)

    def test_follows_the_plans_resampling(self):
        """The validation predictor is built without force_trilinear, so it takes the plans' separate-z path."""
        p = nnUNetPredictorStreamingFromTrainer(device=torch.device('cpu'), allow_tqdm=False)
        self.assertFalse(p.force_trilinear)


class TestResamplingNotice(unittest.TestCase):
    """_log_streaming_validation_notice says whether the plans' resampling is followed or replaced."""

    def _notice(self, fn_name, fn):
        logged = []
        trainer = SimpleNamespace(
            configuration_manager=SimpleNamespace(
                configuration={'resampling_fn_probabilities': fn_name, 'resampling_fn_probabilities_kwargs': {}},
                resampling_fn_probabilities=fn),
            label_manager=SimpleNamespace(num_segmentation_heads=5),
            print_to_log_file=lambda *args, **kwargs: logged.append(' '.join(str(a) for a in args)))
        StreamingValidationMixin._log_streaming_validation_notice(trainer, SimpleNamespace(feature_ring_status='on'))
        return '\n'.join(logged)

    def test_reproducible_resampling_is_followed(self):
        text = self._notice('resample_torch_fornnunet',
                            partial(resample_torch_fornnunet, is_seg=False, force_separate_z=False))
        self.assertIn('Resampling follows the plans', text)
        self.assertNotIn('WARNING', text)

    def test_cubic_resampling_is_replaced(self):
        text = self._notice('resample_data_or_seg_to_shape',
                            partial(resample_data_or_seg_to_shape, is_seg=False, order=3, order_z=0))
        self.assertIn('WARNING', text)


class TestTrainers(unittest.TestCase):
    """The trainers resolve by name and combine their parts as their names say."""

    def _make(self, trainer_class):
        # stands in for nnUNetTrainer.__init__: the trainers only set num_epochs on top of it
        with mock.patch.object(nnUNetTrainer, '__init__', lambda self, *args, **kwargs: None):
            return trainer_class({}, '3d_fullres', 0, {})

    def test_names_resolve(self):
        for name in ('nnUNetTrainer_SV', 'nnUNetTrainer_2000epochs_NoMirroring_SV',
                     'nnUNetTrainer_DASegOrd0_NoMirroring_SV', 'nnUNetTrainer_DASegOrd0_NoMirroring_2000epochs_SV',
                     'nnUNetTrainer_DASegOrd0_NoMirroring_5000epochs_SV'):
            cls = recursive_find_trainer_class_by_name(name)
            self.assertEqual(cls.__name__, name)
            self.assertIs(cls.perform_actual_validation, StreamingValidationMixin.perform_actual_validation)

    def test_many_class_recipe(self):
        for name, epochs in (('nnUNetTrainer_DASegOrd0_NoMirroring_SV', None),
                             ('nnUNetTrainer_DASegOrd0_NoMirroring_2000epochs_SV', 2000),
                             ('nnUNetTrainer_DASegOrd0_NoMirroring_5000epochs_SV', 5000)):
            cls = recursive_find_trainer_class_by_name(name)
            self.assertIs(cls.dice_class, ChunkedMemoryEfficientSoftDiceLoss)
            self.assertIs(cls.dc_and_ce_class, DC_and_CE_loss_chunked)
            self.assertEqual([c.__name__ for c in cls.__mro__ if c.__name__.startswith('nnUNetTrainer_DASegOrd0')][-2:],
                             ['nnUNetTrainer_DASegOrd0_NoMirroring', 'nnUNetTrainer_DASegOrd0'])
            if epochs is not None:
                self.assertEqual(self._make(cls).num_epochs, epochs)

    def test_stock_trainers_keep_the_stock_loss(self):
        for name in ('nnUNetTrainer_SV', 'nnUNetTrainer_2000epochs_NoMirroring_SV'):
            cls = recursive_find_trainer_class_by_name(name)
            self.assertIs(cls.dice_class, nnUNetTrainer.dice_class)
            self.assertIs(cls.dc_and_ce_class, nnUNetTrainer.dc_and_ce_class)
        self.assertEqual(self._make(recursive_find_trainer_class_by_name('nnUNetTrainer_2000epochs_NoMirroring_SV'))
                         .num_epochs, 2000)


if __name__ == '__main__':
    unittest.main()
