"""
nnUNetPredictorStreaming: the ring size, the fused Triton kernels and the standalone-inference settings.

- The ring of patch + 1 slices gives the same segmentation as a much larger ring (CPU and CUDA).
- The fused kernels (accumulate, emit) reproduce the torch path bit for bit, kernel by kernel and end to end on a
  small random network, for an fp16 and an fp32 ring (CUDA + Triton only); kernels_selftest rejects a kernel that is
  off by one rounding or raises, and the predictor then keeps the torch path.
- resolve_compile's precedence (explicit request > nnUNet_compile > available) and set_default_inductor_cache_dir.
"""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.inference import predict_streaming as ps
from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from functools import partial

try:
    import triton  # noqa: F401
    from nnunetv2.inference import streaming_kernels as sk
except ImportError:
    sk = None

CUDA = torch.cuda.is_available()
KERNELS = CUDA and sk is not None
RING_DTYPES = (torch.float16, torch.float32)


class _Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.a = torch.nn.Conv3d(1, 8, 3, padding=1)
        self.b = torch.nn.Conv3d(8, num_classes, 3, padding=1)

    def forward(self, x):
        return self.b(torch.nn.functional.leaky_relu(self.a(x))) * 4


def _predictor(device, num_classes=6, patch=(16, 16, 16), step=0.5, model_spacing=(1., 1., 1.), sepz=False, seed=0):
    torch.manual_seed(seed)
    p = nnUNetPredictorStreaming(tile_step_size=step, use_gaussian=True, use_mirroring=False, device=device,
                                 allow_tqdm=False)
    net = _Net(num_classes)
    p.network = net
    p.list_of_parameters = [net.state_dict()]
    p.allowed_mirroring_axes = None
    p.label_manager = SimpleNamespace(has_regions=False, regions_class_order=None, num_segmentation_heads=num_classes,
                                      foreground_labels=list(range(1, num_classes)))
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


# (source shape, target shape, step, original spacing, model spacing, separate-z plans): down- and upsampling along
# every axis, step 0.5 and 0.8, separate-z along the slab axis and in the cross-section
GEOMETRIES = [
    ((70, 40, 36), (50, 57, 31), 0.5, (1., 1., 1.), (1., 1., 1.), False),
    ((44, 52, 40), (73, 30, 61), 0.8, (1., 1., 1.), (1., 1., 1.), False),
    ((60, 36, 36), (20, 48, 48), 0.5, (3., .75, .75), (1., 1., 1.), True),
    ((36, 64, 36), (36, 21, 50), 0.5, (.75, 3., .75), (1., 1., 1.), True),
]


def _run(p, data, props, margin=None, kernels=True, ring_dtype=torch.float16):
    p._ring_margin = nnUNetPredictorStreaming._ring_margin if margin is None else margin
    p._use_triton_kernels = kernels
    return p._predict_and_export_streaming(data, props, ring_dtype=ring_dtype)


class TestRingAndKernelsEndToEnd(unittest.TestCase):
    def setUp(self):
        self._bench = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True

    def tearDown(self):
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = self._bench

    def _check(self, device, kernels, ring_dtypes=(torch.float16,)):
        for (src, tgt, step, osp, msp, sepz), dt in [(g, d) for g in GEOMETRIES for d in ring_dtypes]:
            with self.subTest(src=src, tgt=tgt, step=step, sepz=sepz, ring_dtype=dt):
                p = _predictor(device, step=step, model_spacing=msp, sepz=sepz)
                data, props = _case(src, tgt, osp)
                big_ring_torch = _run(p, data, props, margin=200, kernels=False, ring_dtype=dt)
                small_ring = _run(p, data, props, kernels=kernels, ring_dtype=dt)
                self.assertEqual(small_ring.shape, tuple(tgt))
                np.testing.assert_array_equal(small_ring, big_ring_torch)
                self.assertGreater(len(np.unique(small_ring)), 2)   # a real argmax, not a constant

    def test_ring_size_cpu(self):
        self._check(torch.device('cpu'), kernels=False, ring_dtypes=RING_DTYPES)

    @unittest.skipUnless(CUDA, 'CUDA')
    def test_ring_size_cuda_torch_path(self):
        self._check(torch.device('cuda'), kernels=False, ring_dtypes=RING_DTYPES)

    @unittest.skipUnless(KERNELS, 'CUDA + Triton')
    def test_kernels_match_torch_path(self):
        self._check(torch.device('cuda'), kernels=True, ring_dtypes=RING_DTYPES)

    @unittest.skipUnless(KERNELS, 'CUDA + Triton')
    def test_kernels_used_for_both_ring_dtypes(self):
        # the match above must not come from a silent fallback to the torch path
        p = _predictor(torch.device('cuda'))
        data, props = _case(*GEOMETRIES[0][:2])
        for dt in RING_DTYPES:
            with self.subTest(ring_dtype=dt), \
                    mock.patch.object(sk, 'emit_argmax', wraps=sk.emit_argmax) as emit, \
                    mock.patch.object(sk, 'accumulate_', wraps=sk.accumulate_) as acc:
                _run(p, data, props, kernels=True, ring_dtype=dt)
                self.assertGreater(emit.call_count, 0)
                self.assertGreater(acc.call_count, 0)
                self.assertEqual(acc.call_args.args[0].dtype, dt)

    def test_ring_guard_fires_when_too_small(self):
        p = _predictor(torch.device('cpu'))
        data, props = _case(*GEOMETRIES[0][:2])
        with self.assertRaises(RuntimeError):
            _run(p, data, props, margin=-4, kernels=False)


def _torch_emit(l0, l1, n0, n1, w, size):
    b0 = F.interpolate((l0 / n0).float()[None], size=size, mode='bilinear', align_corners=False)[0]
    b1 = F.interpolate((l1 / n1).float()[None], size=size, mode='bilinear', align_corners=False)[0]
    return (1.0 - w) * b0 + w * b1


@unittest.skipUnless(KERNELS, 'CUDA + Triton')
class TestKernels(unittest.TestCase):
    def test_accumulate_bitwise(self):
        for dt in RING_DTYPES:
            with self.subTest(ring_dtype=dt):
                self._accumulate_bitwise(dt)

    def _accumulate_bitwise(self, ring_dtype):
        g = torch.Generator(device='cuda').manual_seed(0)
        ring = (torch.randn(5, 30, 40, 50, device='cuda', generator=g) * 3).to(ring_dtype)
        for perm in ((0, 1, 2), (1, 0, 2)):
            pred = (torch.randn(5, 12, 16, 20, device='cuda', generator=g) * 5).half().permute(0, *[1 + i for i in perm])
            gauss = (torch.rand(12, 16, 20, device='cuda', generator=g) * 10).half().permute(*perm)
            sl = (slice(None), slice(3, 3 + pred.shape[1]), slice(5, 5 + pred.shape[2]), slice(7, 7 + pred.shape[3]))
            ref = ring.clone()
            ref[sl] += pred.clone() * gauss
            out = ring.clone()
            sk.accumulate_(out[sl], pred, gauss)
            self.assertTrue(torch.equal(out, ref), perm)

    def test_emit_bitwise_values(self):
        for dt in RING_DTYPES:
            with self.subTest(ring_dtype=dt):
                self._emit_bitwise_values(dt)

    def _emit_bitwise_values(self, ring_dtype):
        g = torch.Generator(device='cuda').manual_seed(1)
        for (C, S0, S1, T0, T1) in ((6, 33, 47, 52, 29), (4, 80, 40, 41, 97), (3, 64, 64, 64, 64)):
            for w, blend in ((0.0, False), (0.25, True), (0.6180339887498949, True)):
                n0 = (torch.rand(S0, S1, device='cuda', generator=g) * 30 + 1).to(ring_dtype)
                n1 = (torch.rand(S0, S1, device='cuda', generator=g) * 30 + 1).to(ring_dtype)
                l0 = (torch.randn(C, S0, S1, device='cuda', generator=g) * 400).to(ring_dtype)
                l1 = (torch.randn(C, S0, S1, device='cuda', generator=g) * 400).to(ring_dtype)
                ref = _torch_emit(l0, l1, n0, n1, w, (T0, T1)) if blend else \
                    F.interpolate((l0 / n0).float()[None], size=(T0, T1), mode='bilinear', align_corners=False)[0]
                ref_val, ref_arg = ref.max(0)
                out = torch.empty((T0, T1), dtype=torch.uint8, device='cuda')
                best = torch.empty((T0, T1), dtype=torch.float32, device='cuda')
                sk.emit_argmax(l0, l1, n0, n1, w, blend, sk.bilinear_tables(S0, T0, 'cuda'),
                               sk.bilinear_tables(S1, T1, 'cuda'), out, best=best)
                self.assertTrue(torch.equal(best, ref_val), (C, S0, S1, T0, T1, w))
                self.assertTrue(torch.equal(out, ref_arg.to(torch.uint8)), (C, S0, S1, T0, T1, w))

    def test_selftest_rejects_one_rounding_off(self):
        for dt in RING_DTYPES:
            self.assertTrue(sk.kernels_selftest(torch.device('cuda'), dt))
        orig = sk.emit_argmax
        try:
            sk._SELFTEST.clear()
            sk.emit_argmax = lambda *a, **k: orig(*a[:4], a[4] + 1e-7, *a[5:], **k)
            for dt in RING_DTYPES:
                self.assertFalse(sk.kernels_selftest(torch.device('cuda'), dt))
        finally:
            sk.emit_argmax = orig
            sk._SELFTEST.clear()

    def test_launches_on_the_device_of_the_tensors(self):
        """Triton launches on the thread's current device: both wrappers make the tensors' device current."""
        entered = []

        def spy(device):
            entered.append(torch.device(device))
            return torch.cuda.device(device)
        sk._SELFTEST.clear()
        try:
            with mock.patch.object(sk, 'cuda_device', side_effect=spy):
                self.assertTrue(sk.kernels_selftest(torch.device('cuda'), torch.float16))
        finally:
            sk._SELFTEST.clear()
        self.assertGreater(len(entered), 1)
        self.assertTrue(all(d.type == 'cuda' and d.index == torch.cuda.current_device() for d in entered), entered)

    @unittest.skipUnless(CUDA and torch.cuda.device_count() > 1, 'two CUDA devices')
    def test_predictor_on_a_device_that_is_not_current(self):
        src, tgt, step, osp, msp, sepz = GEOMETRIES[0]
        data, props = _case(src, tgt, osp)
        previous = torch.cuda.current_device()
        torch.cuda.set_device(0)
        try:
            p = _predictor(torch.device('cuda', 1), step=step, model_spacing=msp, sepz=sepz)
            ref = _run(p, data, props, kernels=False)
            got = _run(p, data, props, kernels=True)
            self.assertEqual(torch.cuda.current_device(), 0)
        finally:
            torch.cuda.set_device(previous)
        np.testing.assert_array_equal(got, ref)

    def test_selftest_rejects_unknown_ring_dtype(self):
        self.assertFalse(sk.kernels_selftest(torch.device('cuda'), torch.bfloat16))

    def test_selftest_survives_kernel_errors(self):
        # e.g. a GPU older than Triton supports: Triton imports, the first kernel compile raises
        orig = sk.accumulate_
        try:
            sk._SELFTEST.clear()
            sk.accumulate_ = mock.Mock(side_effect=RuntimeError('compile failed'))
            self.assertFalse(sk.kernels_selftest(torch.device('cuda')))
        finally:
            sk.accumulate_ = orig
            sk._SELFTEST.clear()

    @unittest.skipUnless(CUDA, 'CUDA')
    def test_predictor_falls_back_when_selftest_fails(self):
        p = _predictor(torch.device('cuda'))
        data, props = _case(*GEOMETRIES[0][:2])
        ref = _run(p, data, props, kernels=False)
        with mock.patch.object(sk, 'kernels_selftest', return_value=False), \
                mock.patch.object(sk, 'emit_argmax', side_effect=AssertionError('kernel used')), \
                mock.patch.object(sk, 'accumulate_', side_effect=AssertionError('kernel used')):
            np.testing.assert_array_equal(_run(p, data, props, kernels=True), ref)


class TestCompileResolution(unittest.TestCase):
    def _resolve(self, requested, env, available):
        environ = {k: v for k, v in os.environ.items() if k != 'nnUNet_compile'}
        if env is not None:
            environ['nnUNet_compile'] = env
        with mock.patch.dict(os.environ, environ, clear=True), \
                mock.patch.object(ps, 'compile_available', return_value=(available, 'test')):
            return ps.resolve_compile(requested, torch.device('cuda'))

    def test_precedence(self):
        # default ('auto'): off without a number of cases; the batch policy is tested in test_predict_streaming_batch
        self.assertFalse(self._resolve(None, None, True))
        self.assertFalse(self._resolve(None, None, False))
        self.assertFalse(self._resolve(None, 'False', True))       # environment variable over the default
        self.assertTrue(self._resolve(None, '1', True))
        self.assertFalse(self._resolve(None, 'true', False))       # ... degrades to a notice where unavailable
        self.assertFalse(self._resolve(False, 'true', True))       # explicit request over everything
        self.assertTrue(self._resolve(True, 'false', True))
        with self.assertRaises(RuntimeError):
            self._resolve(True, None, False)                       # explicit request that cannot be honoured

    def test_unavailable_off_cuda(self):
        self.assertFalse(ps.compile_available(torch.device('cpu'))[0])


class TestInductorCacheDir(unittest.TestCase):
    def test_default_and_override(self):
        from nnunetv2.utilities.helpers import set_default_inductor_cache_dir
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir
        base = {k: v for k, v in os.environ.items() if k not in ('TORCHINDUCTOR_CACHE_DIR', 'XDG_CACHE_HOME')}
        expected = os.path.join('/x', 'nnunetv2', 'torchinductor')
        for preset, result in ((None, expected), (default_cache_dir(), expected), ('/my/cache', '/my/cache')):
            env = dict(base, XDG_CACHE_HOME='/x')
            if preset is not None:
                env['TORCHINDUCTOR_CACHE_DIR'] = preset
            with mock.patch.dict(os.environ, env, clear=True):
                set_default_inductor_cache_dir()
                self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'], result, preset)


if __name__ == '__main__':
    unittest.main()
