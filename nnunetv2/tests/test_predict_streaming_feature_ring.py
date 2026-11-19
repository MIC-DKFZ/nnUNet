"""
nnUNetPredictorStreaming's feature ring: the ring holds the decoder's last features and the detached output layer is
applied per output slice. It must give the logit ring's segmentation up to floating-point rounding.

- single fold on small dynamic_network_architectures UNets (PlainConv, ResEnc), with and without TTA, trilinear and
  separate-z geometries; regions (sigmoid, the unfused emit); 2- and 3-fold ensembles (one feature block per fold) and
  the switch back to the logit ring once folds x features >= classes (CPU, fp32 throughout);
- CUDA: the fused kernels with fp16 and fp32 rings, the network in fp32 (autocast off) and in fp16 (autocast), for
  regions too (the fused region emit gives the torch path's segmentation exactly);
  torch.compile (the layer is detached before compiling, also from a network compiled at initialization);
- detach / restore: the logit-returning nnUNetPredictor methods refuse while the layer is detached, restore_head
  gives back the original network and state dicts; the trainer's validation predictor (heads from the live network)
  restores the trainer's network, compiled or not;
- where it does not apply (2d, deep supervision, unknown architectures) the predictor says why and keeps the logits.
"""
import contextlib
import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet

from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming, find_linear_head, unwrap_network
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet

try:
    import triton  # noqa: F401
    from nnunetv2.inference import streaming_kernels as sk
except ImportError:
    sk = None

CUDA = torch.cuda.is_available()
KERNELS = CUDA and sk is not None
NUM_CLASSES = 13            # > 3 folds x 4 features, < 4 folds x 4 features
FEATURES = 4                # decoder's last features = features_per_stage[0]

# (source shape, target shape, step, original spacing, model spacing, separate-z plans)
GEOMETRIES = [
    ((52, 40, 36), (41, 57, 31), 0.5, (1., 1., 1.), (1., 1., 1.), False),
    ((36, 44, 40), (60, 30, 52), 0.8, (1., 1., 1.), (1., 1., 1.), False),
    ((48, 36, 36), (16, 48, 48), 0.5, (3., .75, .75), (1., 1., 1.), True),
]


def _net(arch='plain', num_classes=NUM_CLASSES, seed=0, deep_supervision=False, dim=3):
    torch.manual_seed(seed)
    conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d
    norm = torch.nn.InstanceNorm3d if dim == 3 else torch.nn.InstanceNorm2d
    common = dict(input_channels=1, n_stages=3, features_per_stage=(FEATURES, 8, 16), conv_op=conv, kernel_sizes=3,
                  strides=(1, 2, 2), num_classes=num_classes, n_conv_per_stage_decoder=1, conv_bias=True,
                  norm_op=norm, norm_op_kwargs={'affine': True, 'eps': 1e-5}, nonlin=torch.nn.LeakyReLU,
                  nonlin_kwargs={'inplace': True}, deep_supervision=deep_supervision)
    if arch == 'plain':
        net = PlainConvUNet(n_conv_per_stage=2, **common)
    else:
        net = ResidualEncoderUNet(n_blocks_per_stage=(1, 1, 1), **common)
    with torch.no_grad():                    # spread the logits so that the argmax has many regions
        head = net.decoder.seg_layers[-1]
        head.weight.mul_(6)
        head.bias.normal_(0, .5)
    return net.eval()


def _state(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _predictor(device, arch='plain', folds=1, num_classes=NUM_CLASSES, step=0.5, model_spacing=(1., 1., 1.),
               sepz=False, tta=False, regions=None, feature_ring=True, patch=(16, 16, 16)):
    p = nnUNetPredictorStreaming(tile_step_size=step, use_gaussian=True, use_mirroring=tta, device=device,
                                 allow_tqdm=False, compile=False, feature_ring=feature_ring)
    nets = [_net(arch, num_classes, seed=s, dim=len(patch)) for s in range(folds)]
    p.network = nets[0]
    p.list_of_parameters = [_state(n) for n in nets]
    p.allowed_mirroring_axes = (0, 1, 2)[:len(patch)] if tta else None
    if regions is None:
        p.label_manager = SimpleNamespace(has_regions=False, regions_class_order=None,
                                          num_segmentation_heads=num_classes,
                                          foreground_labels=list(range(1, num_classes)))
    else:
        p.label_manager = SimpleNamespace(has_regions=True, regions_class_order=list(regions),
                                          num_segmentation_heads=num_classes, foreground_labels=sorted(set(regions)))
    fn = partial(resample_torch_fornnunet, is_seg=False, force_separate_z=None if sepz else False,
                 memefficient_seg_resampling=True)
    p.configuration_manager = SimpleNamespace(patch_size=list(patch), resampling_fn_probabilities=fn,
                                              spacing=list(model_spacing), configuration={})
    p.plans_manager = SimpleNamespace(transpose_forward=[0, 1, 2], transpose_backward=[0, 1, 2])
    return p


def _case(source_shape, target_shape, original_spacing=(1., 1., 1.), seed=1):
    g = torch.Generator().manual_seed(seed)
    low = torch.randn((1, 1, *[max(2, s // 6) for s in source_shape]), generator=g)
    data = F.interpolate(low, size=source_shape, mode='trilinear', align_corners=False)[0].contiguous()
    props = {'shape_after_cropping_and_before_resampling': list(target_shape), 'shape_before_cropping': list(target_shape),
             'bbox_used_for_cropping': [[0, s] for s in target_shape], 'spacing': list(original_spacing)}
    return data, props


@contextlib.contextmanager
def _fp32_network():
    """The predictor's autocast becomes a no-op and TF32 is off: fp32 throughout on CUDA."""
    orig = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with mock.patch.object(torch, 'autocast', lambda *a, **k: contextlib.nullcontext()):
            yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = orig


def _logit_and_feature_ring(make, data, props, ring_dtype=torch.float32, kernels=True):
    """(logit ring, feature ring, feature-ring predictor, output channels of the feature-ring network); one process,
    fixed order."""
    lp = make(feature_ring=False)
    lp._use_triton_kernels = kernels
    lp._prepare_standalone_inference()
    ref = lp._predict_and_export_streaming(data, props, ring_dtype=ring_dtype)
    fp = make(feature_ring=True)
    fp._use_triton_kernels = kernels
    fp._prepare_standalone_inference()
    channels = []
    orig = fp._internal_maybe_mirror_and_predict

    def record(x):
        out = orig(x)
        channels.append(out.shape[1])
        return out
    with mock.patch.object(fp, '_internal_maybe_mirror_and_predict', record):
        got = fp._predict_and_export_streaming(data, props, ring_dtype=ring_dtype)
    return ref, got, fp, set(channels)


class _Base(unittest.TestCase):
    def setUp(self):
        self._cudnn = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

    def tearDown(self):
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = self._cudnn

    def assertSameSegmentation(self, got, ref, min_agreement):
        self.assertEqual(got.shape, ref.shape)
        self.assertGreater(len(np.unique(ref)), 3)          # a real argmax, not a constant
        self.assertGreaterEqual(float(np.mean(got == ref)), min_agreement)


class TestFeatureRingCPU(_Base):
    """fp32 throughout: the two rings differ only in the summation order (W sum g f vs sum g W f)."""
    DEV = torch.device('cpu')

    def test_single_fold(self):
        for arch in ('plain', 'resenc'):
            for tta in (False, True):
                for src, tgt, step, osp, msp, sepz in GEOMETRIES:
                    with self.subTest(arch=arch, tta=tta, src=src, tgt=tgt, sepz=sepz):
                        make = partial(_predictor, self.DEV, arch=arch, step=step, model_spacing=msp, sepz=sepz,
                                       tta=tta)
                        data, props = _case(src, tgt, osp)
                        ref, got, fp, ch = _logit_and_feature_ring(make, data, props)
                        self.assertEqual(ch, {FEATURES})
                        self.assertTrue(fp.feature_ring_status.startswith('on'), fp.feature_ring_status)
                        self.assertSameSegmentation(got, ref, 0.9999)

    def test_regions(self):
        # 4 overlapping regions, painted in this order (the last one wins); thresholds after the linear steps
        make = partial(_predictor, self.DEV, num_classes=6, regions=(1, 2, 3, 4, 5, 6))
        data, props = _case(*GEOMETRIES[0][:2])
        ref, got, fp, ch = _logit_and_feature_ring(make, data, props)
        self.assertEqual(ch, {FEATURES})
        self.assertSameSegmentation(got, ref, 0.9999)

    def test_ensembles(self):
        data, props = _case(*GEOMETRIES[0][:2])
        for folds in (2, 3):
            with self.subTest(folds=folds):
                ref, got, fp, ch = _logit_and_feature_ring(partial(_predictor, self.DEV, folds=folds), data, props)
                self.assertEqual(ch, {FEATURES})
                self.assertEqual(len(fp._detached_head['heads']), folds)
                self.assertSameSegmentation(got, ref, 0.9999)
                # the check must be able to fail: one fold's segmentation is not the ensemble's
                one = _predictor(self.DEV, folds=1, feature_ring=False)._predict_and_export_streaming(
                    data, props, ring_dtype=torch.float32)
                self.assertLess(float(np.mean(one == ref)), 0.99)

    def test_more_features_than_classes_keeps_logit_ring(self):
        data, props = _case(*GEOMETRIES[0][:2])
        ref, got, fp, ch = _logit_and_feature_ring(partial(_predictor, self.DEV, folds=4), data, props)
        self.assertIsNone(fp._detached_head)
        self.assertIn('logit ring is smaller', fp.feature_ring_status)
        self.assertEqual(ch, {NUM_CLASSES})
        np.testing.assert_array_equal(got, ref)

    def test_not_applicable(self):
        # 2d
        p = _predictor(self.DEV, patch=(16, 16))
        p._prepare_standalone_inference()
        self.assertIsNone(p._detached_head)
        self.assertIn('2d', p.feature_ring_status)
        # deep supervision on
        p = _predictor(self.DEV)
        p.network.decoder.deep_supervision = True
        self.assertFalse(p.detach_head())
        self.assertIn('deep supervision', p.feature_ring_status)
        # an architecture whose output layer is unknown

        class Wrapped(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x) * 2
        self.assertIsNone(find_linear_head(Wrapped(_net()))[0])
        p = _predictor(self.DEV)
        p.network = Wrapped(p.network)
        self.assertFalse(p.detach_head())
        self.assertTrue(p.feature_ring_status.startswith('off'))
        # switched off
        p = _predictor(self.DEV, feature_ring=False)
        self.assertFalse(p.detach_head())
        self.assertIn('switched off', p.feature_ring_status)

    def test_detach_restore(self):
        p = _predictor(self.DEV, folds=2)
        x = torch.randn(1, 1, 16, 16, 16)
        full_state = [dict(s) for s in p.list_of_parameters]
        with torch.no_grad():
            before = p.network(x)
        p._prepare_standalone_inference()
        p._prepare_standalone_inference()                        # idempotent: detached once
        self.assertIsInstance(p.network.decoder.seg_layers[-1], torch.nn.Identity)
        self.assertTrue(all('decoder.seg_layers.1.weight' not in s for s in p.list_of_parameters))
        with self.assertRaises(RuntimeError):
            p.predict_sliding_window_return_logits(torch.zeros(1, 16, 16, 16))
        with self.assertRaises(RuntimeError):
            p.predict_logits_from_preprocessed_data(torch.zeros(1, 16, 16, 16))
        data, props = _case(*GEOMETRIES[0][:2])
        p._predict_and_export_streaming(data, props)             # leaves fold 1 loaded
        self.assertIs(p._loaded_parameters[1], p.list_of_parameters[1])
        p.restore_head()
        self.assertIsNone(p._detached_head)
        self.assertEqual([sorted(s) for s in p.list_of_parameters], [sorted(s) for s in full_state])
        # the network is fold 1 again, output layer included
        ref = _net(seed=1)
        with torch.no_grad():
            torch.testing.assert_close(p.network(x), ref(x), rtol=0, atol=0)
        p._load_fold(0)
        with torch.no_grad():
            torch.testing.assert_close(p.network(x), before, rtol=0, atol=0)


@unittest.skipUnless(CUDA, 'CUDA')
class TestFeatureRingCUDA(_Base):
    DEV = torch.device('cuda')

    def _compare(self, ring_dtype, fp32_net, kernels, min_agreement, **kw):
        for src, tgt, step, osp, msp, sepz in GEOMETRIES:
            with self.subTest(src=src, tgt=tgt, sepz=sepz, ring_dtype=ring_dtype, fp32_net=fp32_net, **kw):
                make = partial(_predictor, self.DEV, step=step, model_spacing=msp, sepz=sepz, **kw)
                data, props = _case(src, tgt, osp)
                with (_fp32_network() if fp32_net else contextlib.nullcontext()):
                    ref, got, fp, ch = _logit_and_feature_ring(make, data, props, ring_dtype, kernels)
                self.assertEqual(ch, {FEATURES})
                self.assertSameSegmentation(got, ref, min_agreement)

    def test_fp32_throughout(self):
        for kernels in ((False, True) if KERNELS else (False,)):
            self._compare(torch.float32, True, kernels, 0.9999, folds=2, tta=True)

    def test_autocast_fp16_and_fp32_rings(self):
        # the network's own fp16 rounding is in both arms; what differs is where the output layer runs (fp16 inside
        # the network vs fp32 per slice) and the ring's rounding
        for dt in (torch.float16, torch.float32):
            self._compare(dt, False, KERNELS, 0.998, tta=True)

    def test_regions(self):
        # the feature ring's fp32 slice logits go through the region threshold, fused (emit_regions) or in torch
        for kernels in ((False, True) if KERNELS else (False,)):
            self._compare(torch.float32, True, kernels, 0.9999, num_classes=6, regions=(1, 2, 3, 4, 5, 6))

    @unittest.skipUnless(KERNELS, 'CUDA + Triton')
    def test_regions_fused_emit_matches_torch_path(self):
        make = partial(_predictor, self.DEV, num_classes=6, regions=(3, 1, 4, 1, 5, 2))
        data, props = _case(*GEOMETRIES[0][:2])
        segs = []
        for kernels in (False, True):
            p = make()
            p._use_triton_kernels = kernels
            p._prepare_standalone_inference()
            self.assertIsNotNone(p._detached_head)
            with mock.patch.object(sk, 'emit_regions', wraps=sk.emit_regions) as emit:
                segs.append(p._predict_and_export_streaming(data, props))
            self.assertEqual(emit.call_count > 0, kernels)
        self.assertGreater(len(np.unique(segs[0])), 3)
        np.testing.assert_array_equal(segs[1], segs[0])

    @unittest.skipUnless(KERNELS, 'CUDA + Triton')
    def test_fused_kernels_used(self):
        p = _predictor(self.DEV, folds=2)
        p._prepare_standalone_inference()
        data, props = _case(*GEOMETRIES[0][:2])
        for dt in (torch.float16, torch.float32):
            with self.subTest(ring_dtype=dt), \
                    mock.patch.object(sk, 'emit_argmax', wraps=sk.emit_argmax) as emit, \
                    mock.patch.object(sk, 'accumulate_', wraps=sk.accumulate_) as acc:
                p._predict_and_export_streaming(data, props, ring_dtype=dt)
                self.assertGreater(emit.call_count, 0)
                self.assertGreater(acc.call_count, 0)
                self.assertEqual(acc.call_args.args[0].shape[0], FEATURES)     # one fold's block of the ring
                self.assertEqual(acc.call_args.args[0].dtype, dt)
                self.assertEqual(emit.call_args.args[0].dtype, torch.float32)  # logits of the slice, fp32

    @unittest.skipUnless(KERNELS, 'CUDA + Triton')
    def test_compile(self):
        data, props = _case(*GEOMETRIES[0][:2])
        ref = _predictor(self.DEV, feature_ring=False)._predict_and_export_streaming(data, props)
        for precompiled in (False, True):
            with self.subTest(precompiled=precompiled):
                p = _predictor(self.DEV)
                p.compile = True
                if precompiled:                                   # as initialize_* does with nnUNet_compile=True
                    p.network = torch.compile(p.network)
                p._prepare_standalone_inference()
                self.assertTrue(hasattr(p.network, '_orig_mod'))
                self.assertIsInstance(unwrap_network(p.network).decoder.seg_layers[-1], torch.nn.Identity)
                got = p._predict_and_export_streaming(data, props)
                self.assertSameSegmentation(got, ref, 0.998)


if __name__ == '__main__':
    unittest.main()
