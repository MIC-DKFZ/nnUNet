"""
Probability export of nnUNetPredictorStreaming (save_probabilities, streaming_probabilities.py).

- StreamingProbabilityWriter alone, fed the planes of a known logit volume along each slab axis: the file must equal
  nnU-Net's own export of those probabilities (nonlinearity, revert_cropping_on_probabilities, transpose_backward),
  quantized, for cropped cases, transposed plans, chunk rows that do not divide the image, region models and float32;
  on CPU and on CUDA (device staging rows, pinned host rows, side-stream copies).
- End to end against nnUNetPredictor (predict_single_npy_array with probabilities; its logits accumulated in float32
  here, against streaming's float32 ring) with tiny model folders: softmax models (cropped and transposed cases,
  float32 files), region models (nested, overlapping, ignore label, 2d, separate-z + 2 folds + TTA; CPU and CUDA), the
  segmentation with and without export, files and --continue_prediction. (With nnUNetPredictor's float16 accumulator
  these fixtures' deliberately large logits differ by up to 0.1-0.5 in single voxels between any two summation
  orders, streaming's float16 ring included.)
"""
import os
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, load_pickle, maybe_mkdir_p, \
    save_json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming, predict_entry_point_streaming
from nnunetv2.inference.streaming_pipeline import load_npy_case
from nnunetv2.inference.streaming_probabilities import PROBABILITY_SCALE, StreamingProbabilityWriter, \
    load_streamed_probabilities, open_streamed_probabilities
from nnunetv2.tests.test_predict_streaming_batch import NUM_CLASSES, write_model_folder
from nnunetv2.tests.test_predict_streaming_feature_ring import _case, _fp32_network, _predictor
from nnunetv2.tests.test_predict_streaming_regions import NESTED, OVERLAPPING, WITH_IGNORE, write_region_model
from nnunetv2.tests.test_predict_streaming_regions import _image as _smooth_image
from nnunetv2.utilities.label_handling.label_handling import LabelManager

CPU = torch.device('cpu')
SOFTMAX = LabelManager({'background': 0, 'a': 1, 'b': 2, 'c': 3}, None)
FULL, BBOX = (23, 19, 21), [[3, 20], [0, 19], [5, 16]]           # cropped along two axes, not along the third


def _image(seed, shape, zero_border=0):
    """A smooth random image; with zero_border its border is 0, so nnU-Net crops it away."""
    img = _smooth_image(seed, shape)
    if zero_border:
        b = zero_border
        mask = np.zeros_like(img, dtype=bool)
        mask[:, b:-b, :, b + 2:-b] = True
        img[~mask] = 0
    return img


def return_segmentation(segmentation, properties, output):
    """export_fn for npy cases (no reader properties to write a file with): hand the segmentation back."""
    return segmentation


def _eager(test: unittest.TestCase):
    """Both predictors eager (nnUNet_compile may be set in the environment) and deterministic cuDNN, for this test."""
    env = mock.patch.dict(os.environ, {'nnUNet_compile': 'false'})
    env.start()
    test.addCleanup(env.stop)
    cudnn = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    test.addCleanup(lambda: setattr(torch.backends.cudnn, 'benchmark', cudnn[0]))
    test.addCleanup(lambda: setattr(torch.backends.cudnn, 'deterministic', cudnn[1]))


def _stock(model, device, folds=(0,)) -> nnUNetPredictor:
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                        perform_everything_on_device=device.type == 'cuda', device=device, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model, tuple(folds), 'checkpoint_final.safetensors')
    return p


def _streaming(test, model, device, folds=(0,)) -> nnUNetPredictorStreaming:
    p = nnUNetPredictorStreaming(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, device=device,
                                 allow_tqdm=False, quiet=True, compile=False)
    p.initialize_from_folder_base_trainer(model, list(folds))
    test.addCleanup(p.close)
    return p


def _compare(stock, p, img, props, out, slab_axis=None):
    """Stock probabilities (float32 accumulator) and the streamed ones (float32 ring, written to out + '.b2nd'):
    (stock segmentation, stock probabilities, streamed segmentation, streamed probabilities)."""
    with mock.patch.object(torch, 'half', torch.float32):              # nnUNetPredictor's logit accumulator
        seg_ref, prob_ref = stock.predict_single_npy_array(img, dict(props), None, None, True)
    seg = p.predict_single_case_streaming((img, dict(props)), load_npy_case, out, return_segmentation,
                                          slab_axis=slab_axis, ring_dtype=torch.float32, save_probabilities=True)
    return seg_ref, prob_ref, seg, load_streamed_probabilities(out + '.b2nd')


def _reference(logits: torch.Tensor, lm: LabelManager, properties: dict, transpose_backward) -> np.ndarray:
    """nnU-Net's export of these (target-grid, plans' axis order) logits: probabilities in the reader's geometry."""
    probs = lm.apply_inference_nonlin(logits)
    probs = lm.revert_cropping_on_probabilities(probs, properties['bbox_used_for_cropping'],
                                                properties['shape_before_cropping'])
    return probs.numpy().transpose([0] + [i + 1 for i in transpose_backward])


class TestWriter(unittest.TestCase):
    DEV = CPU

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _run(self, lm, full, bbox, transpose_backward, slab_axis, K, dtype='uint16'):
        C = lm.num_segmentation_heads
        target = [hi - lo for lo, hi in bbox]
        logits = torch.randn((C, *target), generator=torch.Generator().manual_seed(0)) * 4
        props = {'shape_before_cropping': list(full), 'bbox_used_for_cropping': [list(b) for b in bbox],
                 'shape_after_cropping_and_before_resampling': target, 'spacing': [1., 1., 1.]}
        out = join(self._tmp.name, f'case_{slab_axis}_{K}_{dtype}')
        w = StreamingProbabilityWriter(out, lm, props, transpose_backward, self.DEV, planes_per_chunk=K,
                                       compression_threads=2, dtype=dtype)
        w.begin(slab_axis)
        cross = [ax for ax in range(3) if ax != slab_axis]
        canonical = logits.permute(0, 1 + slab_axis, 1 + cross[0], 1 + cross[1])
        for j in range(target[slab_axis]):
            w.put(j, canonical[:, j].to(self.DEV, copy=True))
        w.close()
        self.assertFalse(os.path.exists(w.tmp_path))
        self.assertEqual(load_pickle(out + '.pkl'), props)
        arr = open_streamed_probabilities(out + '.b2nd')
        ref = _reference(logits, lm, props, transpose_backward)
        self.assertEqual(tuple(arr.shape), ref.shape)
        self.assertEqual(arr.dtype, np.dtype(dtype))
        meta = arr.schunk.vlmeta['nnunet']
        self.assertEqual((meta['nonlinearity'], meta['dtype']), ('sigmoid' if lm.has_regions else 'softmax', dtype))
        if dtype == 'uint16':
            # per-plane vs whole-volume nonlinearity may differ in the last float bit: at most one quantization step
            quantized = np.rint(ref.astype(np.float64) * PROBABILITY_SCALE).astype(np.int64)
            self.assertLessEqual(np.abs(arr[:].astype(np.int64) - quantized).max(), 1)
        atol = 1.0 / PROBABILITY_SCALE if dtype == 'uint16' else 1e-6
        np.testing.assert_allclose(load_streamed_probabilities(out + '.b2nd'), ref, atol=atol, rtol=0)
        np.testing.assert_allclose(load_streamed_probabilities(out + '.b2nd', channels=1), ref[1], atol=atol, rtol=0)

    def test_geometry_softmax(self):
        for tb in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
            for a in range(3):
                for K in (1, 4, 6):
                    with self.subTest(transpose_backward=tb, slab_axis=a, K=K):
                        self._run(SOFTMAX, FULL, BBOX, tb, a, K)

    def test_float32(self):
        for tb, a in (([0, 1, 2], 0), ([2, 0, 1], 2)):
            with self.subTest(transpose_backward=tb, slab_axis=a), \
                    self.assertWarnsRegex(UserWarning, 'hardly compress'):
                self._run(SOFTMAX, FULL, BBOX, tb, a, 4, dtype='float32')

    def test_regions(self):
        lm = LabelManager({'background': 0, 'whole': [1, 2, 3], 'core': [2, 3], 'enh': 3}, [1, 2, 3])
        self.assertTrue(lm.has_regions)
        for dtype in ('uint16', 'float32'):
            with self.subTest(dtype=dtype):
                self._run(lm, (17, 20, 15), [[2, 15], [1, 19], [0, 15]], [0, 2, 1], 1, 4, dtype=dtype)

    def test_unknown_dtype(self):
        for dtype in ('float16', 'float64'):
            with self.assertRaises(ValueError):
                StreamingProbabilityWriter(join(self._tmp.name, 'x'), SOFTMAX, {}, [0, 1, 2], CPU, dtype=dtype)

    def test_abort_leaves_nothing(self):
        props = {'shape_before_cropping': [8, 8, 8], 'bbox_used_for_cropping': [[0, 8]] * 3}
        w = StreamingProbabilityWriter(join(self._tmp.name, 'aborted'), SOFTMAX, props, [0, 1, 2], self.DEV, 4, 1)
        w.begin(0)
        w.put(0, torch.zeros((4, 8, 8), device=self.DEV))
        w.abort()
        self.assertEqual(os.listdir(self._tmp.name), [])


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA')
class TestWriterCUDA(TestWriter):
    """The same on the GPU: device staging rows, pinned host rows, side-stream copies and their events."""
    DEV = torch.device('cuda')


class TestTwoD(unittest.TestCase):
    def test_2d_configuration(self):
        p = _predictor(CPU, patch=(16, 16), feature_ring=False, num_classes=13, model_spacing=(1., 1.))
        p.label_manager = LabelManager({'background': 0, **{f'c{i}': i for i in range(1, 13)}}, None)
        data, props = _case((7, 30, 26), (7, 37, 21))
        props['shape_before_cropping'] = [9, 40, 21]
        props['bbox_used_for_cropping'] = [[1, 8], [2, 39], [0, 21]]
        with tempfile.TemporaryDirectory() as tmp:
            seg = p._predict_case(data, props, None, torch.float32, join(tmp, 'case2d'))
            probs = load_streamed_probabilities(join(tmp, 'case2d.b2nd'))
        self.assertEqual(probs.shape, (13, 9, 40, 21))
        self.assertEqual(seg.shape, (9, 40, 21))
        np.testing.assert_allclose(probs.sum(0), 1, atol=13 * 0.5 / PROBABILITY_SCALE)
        self.assertGreater((probs.argmax(0) == seg).mean(), 0.999)
        self.assertTrue(np.all(probs[0, 0] == 1) and np.all(probs[0, :, :2] == 1))       # outside the crop
        self.assertGreater(len(np.unique(seg[1:8, 2:39])), 4)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = join(cls._tmp.name, 'model')
        write_model_folder(cls.model)
        # the same model with transposed plans
        cls.model_t = join(cls._tmp.name, 'model_t')
        write_model_folder(cls.model_t)
        plans = load_json(join(cls.model_t, 'plans.json'))
        plans['transpose_forward'], plans['transpose_backward'] = [1, 2, 0], [2, 0, 1]
        save_json(plans, join(cls.model_t, 'plans.json'))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        _eager(self)

    def test_matches_stock_predictor(self):
        cases = [((36, 30, 33), (1.3, 0.8, 1.0), 0), ((28, 40, 31), (1.0, 1.0, 1.0), 4)]
        for model in (self.model, self.model_t):
            stock, p = _stock(model, CPU), _streaming(self, model, CPU)
            for k, (shape, spacing, border) in enumerate(cases):
                for slab_axis in (None, 1):
                    with self.subTest(model=os.path.basename(model), case=k, slab_axis=slab_axis):
                        img, props = _image(k, shape, border), {'spacing': list(spacing)}
                        out = join(self._tmp.name, f'e2e_{os.path.basename(model)}_{k}_{slab_axis}')
                        seg_ref, prob_ref, seg, got = _compare(stock, p, img, props, out, slab_axis)
                        self.assertEqual(got.shape, prob_ref.shape)
                        diff = np.abs(got - prob_ref)
                        self.assertLess(diff.max(), 1e-4)                  # quantization step 1.5e-5 + float32 order
                        self.assertLess(diff.mean(), 1e-5)
                        self.assertTrue(os.path.isfile(out + '.pkl'))
                        seg_plain = p.predict_single_npy_array_streaming(img, dict(props), None, slab_axis=slab_axis,
                                                                         ring_dtype=torch.float32)
                        np.testing.assert_array_equal(seg, seg_plain)          # unchanged by the export
                        self.assertGreater((seg_plain == seg_ref).mean(), 0.99)
                        self.assertGreater((got.argmax(0) == seg_plain).mean(), 0.999)

    def test_float32_matches_stock_predictor(self):
        stock, p = _stock(self.model, CPU), _streaming(self, self.model, CPU)
        p.probability_dtype = 'float32'
        with self.assertWarnsRegex(UserWarning, 'hardly compress'):
            _, prob_ref, _, got = _compare(stock, p, _image(5, (36, 30, 33), 3), {'spacing': [1.3, 0.8, 1.0]},
                                           join(self._tmp.name, 'float32'))
        self.assertEqual(open_streamed_probabilities(join(self._tmp.name, 'float32.b2nd')).dtype, np.float32)
        self.assertLess(np.abs(got - prob_ref).max(), 1e-5)              # float32 summation order only

    def test_cli_float32_needs_save_probabilities(self):
        argv = ['nnUNetv2_predict_streaming', '-i', 'in', '-o', 'out', '-d', '1', '-c', '3d_fullres', '--float32']
        with mock.patch('sys.argv', argv), mock.patch('sys.stderr'), mock.patch('builtins.print'), \
                self.assertRaises(SystemExit):
            predict_entry_point_streaming()

    def test_checked_before_predicting(self):
        p = _streaming(self, self.model, CPU)
        img, props = _image(0, (20, 20, 20)), {'spacing': [1., 1., 1.]}
        with self.assertRaises(ValueError):                                 # no output to write next to
            p.predict_single_npy_array_streaming(img, dict(props), None, save_probabilities=True)
        p.probability_dtype = 'float16'
        with self.assertRaises(ValueError), mock.patch.object(p, '_predict_and_export_streaming') as predict:
            p.predict_cases_streaming([(img, props)] * 2, load_npy_case, ['a', 'b'], on_error='skip',
                                      save_probabilities=True)
        predict.assert_not_called()

    def test_files_and_continue(self):
        p = _streaming(self, self.model, CPU)
        src, out = join(self._tmp.name, 'files_in'), join(self._tmp.name, 'files_out')
        maybe_mkdir_p(src)
        for k, shape in enumerate([(30, 26, 28), (24, 30, 27), (27, 25, 31)]):
            nib.save(nib.Nifti1Image(_image(k, shape)[0], np.eye(4)), join(src, f'case_{k}_0000.nii.gz'))
        p.predict_from_files_streaming(src, out, save_probabilities=True)
        for k in range(3):
            for ending in ('.nii.gz', '.b2nd', '.pkl'):
                self.assertTrue(os.path.isfile(join(out, f'case_{k}{ending}')), ending)
            # the reader's array order, as nnUNetv2_predict's .npz (NibabelIOWithReorient: nibabel's axes reversed)
            seg = np.asanyarray(nib.load(join(out, f'case_{k}.nii.gz')).dataobj).transpose(2, 1, 0)
            probs = load_streamed_probabilities(join(out, f'case_{k}.b2nd'))
            self.assertEqual(probs.shape, (NUM_CLASSES, *seg.shape))
            self.assertGreater((probs.argmax(0) == seg).mean(), 0.999)
        self.assertFalse([f for f in os.listdir(out) if 'partial' in f])
        # --continue_prediction: a case without probabilities is predicted again, the others are not
        os.remove(join(out, 'case_1.b2nd'))
        mtimes = {k: os.path.getmtime(join(out, f'case_{k}.nii.gz')) for k in (0, 2)}
        p.predict_from_files_streaming(src, out, overwrite=False, save_probabilities=True)
        self.assertTrue(os.path.isfile(join(out, 'case_1.b2nd')))
        self.assertEqual({k: os.path.getmtime(join(out, f'case_{k}.nii.gz')) for k in (0, 2)}, mtimes)


class TestRegionsEndToEnd(unittest.TestCase):
    """
    Region-based (sigmoid) models end to end, with the region model fixture of test_predict_streaming_regions. Per case:
    the streamed sigmoid probabilities match nnUNetPredictor's, the segmentation is the same with and without export
    (on CUDA: fused region emit vs the torch emit the export uses), nnU-Net's region thresholding of the stored
    probabilities gives that segmentation back (but for probabilities within one quantization step of 0.5), and outside
    the crop the probabilities are 0. On CUDA the network runs in float32 (_fp32_network): under autocast, voxels that
    only the edges of the tiles' Gaussian weighting cover differ between any two predictors (the float16 weights
    underflow there, as in the segmentation), which is not what this test is about.
    """
    DEV = CPU

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        _eager(self)
        if self.DEV.type == 'cuda':
            fp32 = _fp32_network()
            fp32.__enter__()
            self.addCleanup(fp32.__exit__, None, None, None)

    def _check(self, name, label_def, cases, **model_kw):
        model = join(self._tmp.name, name)
        write_region_model(model, label_def, **model_kw)
        folds = range(model_kw.get('folds', 1))
        stock, p = _stock(model, self.DEV, folds), _streaming(self, model, self.DEV, folds)
        lm = p.label_manager
        self.assertTrue(lm.has_regions)
        for k, (shape, spacing, border) in enumerate(cases):
            with self.subTest(model=name, case=k):
                img, props = _image(20 + k, shape, border), {'spacing': list(spacing)}
                seg_ref, prob_ref, seg, got = _compare(stock, p, img, props, join(self._tmp.name, f'{name}_{k}'))
                seg_plain = p.predict_single_npy_array_streaming(img, dict(props), None, ring_dtype=torch.float32)
                np.testing.assert_array_equal(seg, seg_plain)
                self.assertEqual(got.shape, prob_ref.shape)                     # one channel per region
                diff = np.abs(got - prob_ref)
                self.assertLess(diff.max(), 1e-4)
                self.assertLess(diff.mean(), 1e-5)
                self.assertGreater((seg == seg_ref).mean(), 0.99)
                from_probs = np.asarray(lm.convert_probabilities_to_segmentation(got))
                near_half = (np.abs(got - 0.5) <= 1.0 / PROBABILITY_SCALE).any(0)
                np.testing.assert_array_equal(from_probs[~near_half], seg[~near_half])
                if border:
                    self.assertTrue(np.all(got[:, :border] == 0))

    def test_layouts_3d(self):
        cases = [((36, 30, 33), (1.3, 0.8, 1.0), 0), ((28, 40, 31), (1.0, 1.0, 1.0), 4)]
        for name, label_def in (('nested', NESTED), ('overlapping', OVERLAPPING), ('ignore', WITH_IGNORE)):
            self._check(name, label_def, cases)

    def test_2d(self):
        self._check('overlapping_2d', OVERLAPPING,
                    [((9, 40, 37), (1.0, 1.0, 1.0), 0), ((7, 33, 45), (3.0, 1.2, 0.8), 3)], dim=2)

    def test_separate_z_folds_tta(self):
        # 4 x 0.9 x 0.9 mm: separate-z fires (the default plans' scipy resampling, force_separate_z None)
        self._check('nested_sepz', NESTED, [((10, 40, 36), (4.0, 0.9, 0.9), 0), ((30, 28, 34), (1.0, 1.0, 1.0), 3)],
                    sepz=True, folds=2, tta=True)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA')
class TestRegionsEndToEndCUDA(TestRegionsEndToEnd):
    """The same on the GPU: the CUDA writer, and the fused region emit when no probabilities are exported."""
    DEV = torch.device('cuda')


if __name__ == '__main__':
    unittest.main()
