"""
nnUNetPredictorStreaming's batch machinery (streaming_pipeline.py): persistent worker pools, pluggable load_fn /
export_fn, the single-case in-process path, background weight loading, the compile policy, quiet mode.

All on CPU with a tiny model folder written by hand (plans.json, dataset.json, a .safetensors checkpoint), so the
preprocessing workers rebuild the plans exactly as they do for a trained model. Cases have different shapes and
spacings, so both resampling steps run. The core streaming pass is covered by the other test_predict_streaming_* files;
here every comparison is "batch through the pools" against "the same cases one by one in-process": they must be
identical voxel for voxel (CPU is deterministic).
"""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json

import nnunetv2.inference.predict_streaming as ps
from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming
from nnunetv2.inference.streaming_pipeline import CaseFailure
from nnunetv2.utilities.checkpoints import save_checkpoint_file
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

NUM_CLASSES = 6
CPU = torch.device('cpu')

# (seed, shape, spacing, tag): different shapes and spacings -> both resampling steps run (model spacing 1 mm)
CASES = [(1, (36, 30, 33), (1.3, 0.8, 1.0), 'a'),
         (2, (28, 40, 31), (1.0, 1.0, 1.0), 'b'),
         (3, (41, 33, 29), (0.9, 1.2, 1.5), 'c')]


# ------------------------------------------------------------------ module level: picklable for the workers --------
def _image(seed, shape):
    g = torch.Generator().manual_seed(seed)
    low = torch.randn((1, 1, *[max(2, s // 5) for s in shape]), generator=g)
    return F.interpolate(low, size=shape, mode='trilinear', align_corners=False)[0].numpy()


def synthetic_load(case):
    seed, shape, spacing, tag = case
    if tag == 'bad':
        raise ValueError('cannot load this case')
    if tag == 'die':
        os._exit(3)                         # what the OS killing a worker for RAM looks like to the pool
    return _image(seed, shape), {'spacing': list(spacing), 'tag': tag}


def summary_export(segmentation, properties, output, scale):
    if properties['tag'] == 'bad_export':
        raise OSError('cannot write this case')
    return {'output': output, 'tag': properties['tag'], 'scale': scale, 'seg': segmentation}


# ------------------------------------------------------------------ fixture -----------------------------------------
def _plans(patch=(16, 16, 16)):
    arch = {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
            'arch_kwargs': {'n_stages': 3, 'features_per_stage': [4, 8, 16],
                            'conv_op': 'torch.nn.modules.conv.Conv3d', 'kernel_sizes': [[3, 3, 3]] * 3,
                            'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]], 'n_conv_per_stage': [2, 2, 2],
                            'n_conv_per_stage_decoder': [1, 1], 'conv_bias': True,
                            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
                            'norm_op_kwargs': {'eps': 1e-5, 'affine': True}, 'dropout_op': None,
                            'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU',
                            'nonlin_kwargs': {'inplace': True}},
            '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}
    rs = {'is_seg': False, 'force_separate_z': False, 'memefficient_seg_resampling': True}
    cfg = {'data_identifier': 'test_3d_fullres', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 2,
           'patch_size': list(patch), 'median_image_size_in_voxels': [40, 40, 40], 'spacing': [1., 1., 1.],
           'normalization_schemes': ['ZScoreNormalization'], 'use_mask_for_norm': [False],
           'resampling_fn_data': 'resample_torch_fornnunet', 'resampling_fn_seg': 'resample_torch_fornnunet',
           'resampling_fn_data_kwargs': rs, 'resampling_fn_seg_kwargs': dict(rs, is_seg=True),
           'resampling_fn_probabilities': 'resample_torch_fornnunet', 'resampling_fn_probabilities_kwargs': rs,
           'architecture': arch, 'batch_dice': False}
    return {'dataset_name': 'Dataset999_StreamingBatchTest', 'plans_name': 'nnUNetPlans',
            'original_median_spacing_after_transp': [1., 1., 1.], 'original_median_shape_after_transp': [40, 40, 40],
            'image_reader_writer': 'NibabelIOWithReorient', 'transpose_forward': [0, 1, 2],
            'transpose_backward': [0, 1, 2], 'configurations': {'3d_fullres': cfg}, 'experiment_planner_used': 'test',
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': {'0': {'max': 3., 'mean': 0., 'median': 0., 'min': -3.,
                                                                  'percentile_00_5': -2.5, 'percentile_99_5': 2.5,
                                                                  'std': 1.}}}


def write_model_folder(folder: str, seed: int = 0):
    plans = _plans()
    dataset_json = {'channel_names': {'0': 'noise'}, 'labels': {'background': 0, **{f'c{i}': i for i in
                                                                                    range(1, NUM_CLASSES)}},
                    'numTraining': 1, 'file_ending': '.nii.gz'}
    arch = plans['configurations']['3d_fullres']['architecture']
    torch.manual_seed(seed)
    net = get_network_from_plans(arch['network_class_name'], arch['arch_kwargs'], arch['_kw_requires_import'], 1,
                                 NUM_CLASSES, allow_init=True, deep_supervision=False)
    with torch.no_grad():                    # spread the logits so that the argmax has many regions
        head = net.decoder.seg_layers[-1]
        head.weight.mul_(6)
        head.bias.normal_(0, .5)
    maybe_mkdir_p(join(folder, 'fold_0'))
    save_json(plans, join(folder, 'plans.json'))
    save_json(dataset_json, join(folder, 'dataset.json'))
    save_checkpoint_file({'network_weights': net.state_dict(), 'trainer_name': 'nnUNetTrainer',
                          'init_args': {'configuration': '3d_fullres', 'fold': 0},
                          'inference_allowed_mirroring_axes': None},
                         join(folder, 'fold_0', 'checkpoint_final.safetensors'))


class TestStreamingBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = join(cls._tmp.name, 'model')
        write_model_folder(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def predictor(self, background=False, **kwargs) -> nnUNetPredictorStreaming:
        kwargs.setdefault('quiet', True)
        p = nnUNetPredictorStreaming(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, device=CPU,
                                     allow_tqdm=False, num_processes_preprocessing=2, num_processes_export=2, **kwargs)
        p.initialize_from_folder_base_trainer(self.model, [0], load_weights_in_background=background)
        self.addCleanup(p.close)
        return p

    def in_process(self, p, cases=CASES):
        return [p.predict_single_case_streaming(c, synthetic_load, f'out_{c[3]}', summary_export, {'scale': 2})
                for c in cases]

    def assertSameResults(self, got, ref):
        self.assertEqual(len(got), len(ref))
        for g, r in zip(got, ref):
            self.assertEqual((g['output'], g['tag'], g['scale']), (r['output'], r['tag'], r['scale']))
            np.testing.assert_array_equal(g['seg'], r['seg'])

    def test_batch_equals_in_process(self):
        p = self.predictor()
        ref = self.in_process(p)
        for r, c in zip(ref, CASES):
            self.assertEqual(r['seg'].shape, c[1])               # back in the input geometry
            self.assertGreater(len(np.unique(r['seg'])), 2)
        got = p.predict_cases_streaming(CASES, synthetic_load, [f'out_{c[3]}' for c in CASES], summary_export,
                                        {'scale': 2})
        self.assertSameResults(got, ref)

    def test_npy_wrappers(self):
        p = self.predictor()
        images = [_image(c[0], c[1]) for c in CASES]
        props = [{'spacing': list(c[2])} for c in CASES]
        ref = [p.predict_single_npy_array_streaming(i, pr) for i, pr in zip(images, props)]
        got = p.predict_from_list_of_npy_arrays_streaming(images, props)
        for g, r in zip(got, ref):
            np.testing.assert_array_equal(g, r)
        self.assertEqual(props[0], {'spacing': list(CASES[0][2])})  # the caller's properties are not modified

    def test_pools_persist_across_calls(self):
        p = self.predictor()
        p.predict_cases_streaming(CASES, synthetic_load, None, summary_export, {'scale': 1})
        pools = {k: v[1] for k, v in p._pools.items()}
        self.assertEqual(set(pools), {'preprocessing', 'export'})
        pids = {k: set(v._processes) for k, v in pools.items()}
        p.predict_cases_streaming(CASES[:2], synthetic_load, None, summary_export, {'scale': 1})
        self.assertEqual({k: v[1] for k, v in p._pools.items()}, pools)        # same pools ...
        self.assertTrue(all(pids[k] <= set(v._processes) for k, v in pools.items()))   # ... same workers
        p.close()
        self.assertEqual(p._pools, {})
        got = p.predict_cases_streaming(CASES, synthetic_load, [f'out_{c[3]}' for c in CASES], summary_export,
                                        {'scale': 2})
        self.assertSameResults(got, self.in_process(p))                       # fresh pools after close()

    def test_single_case_needs_no_pool(self):
        p = self.predictor()
        p.predict_cases_streaming(CASES[:1], synthetic_load, None, summary_export, {'scale': 1})
        self.assertEqual(p._pools, {})

    def test_files(self):
        p = self.predictor()
        src, out_batch, out_single = (join(self._tmp.name, f'files_{n}') for n in ('in', 'batch', 'single'))
        for d in (src, out_batch, out_single):
            maybe_mkdir_p(d)
        for seed, shape, spacing, tag in CASES:
            affine = np.diag([*spacing, 1.])
            affine[0, 0] *= -1                                  # not RAS: the reader reorients, the writer reverts
            nib.save(nib.Nifti1Image(_image(seed, shape)[0], affine), join(src, f'case_{tag}_0000.nii.gz'))
        p.predict_from_files_streaming(src, out_batch)
        for _, _, _, tag in CASES:
            p.predict_single_case_streaming([join(src, f'case_{tag}_0000.nii.gz')], None, join(out_single, f'case_{tag}'))
        for _, shape, _, tag in CASES:
            a = nib.load(join(out_batch, f'case_{tag}.nii.gz'))
            b = nib.load(join(out_single, f'case_{tag}.nii.gz'))
            self.assertEqual(a.shape, shape)
            np.testing.assert_array_equal(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj))
            np.testing.assert_allclose(a.affine, b.affine)

    def test_background_weights(self):
        ref = self.in_process(self.predictor())
        p = self.predictor(background=True)
        self.assertIsNotNone(p._weights_loader)
        self.assertIsNone(p.list_of_parameters)
        self.assertSameResults(self.in_process(p), ref)
        self.assertIsNone(p._weights_loader)
        p = self.predictor(background=True)
        self.assertSameResults(p.predict_cases_streaming(CASES, synthetic_load, [f'out_{c[3]}' for c in CASES],
                                                         summary_export, {'scale': 2}), ref)

    def test_worker_exception_propagates_pool_survives(self):
        p = self.predictor()
        with self.assertRaises(ValueError):
            p.predict_cases_streaming([*CASES[:2], (9, (20, 20, 20), (1., 1., 1.), 'bad')], synthetic_load)
        pool = p._pools['preprocessing'][1]
        got = p.predict_cases_streaming(CASES, synthetic_load, [f'out_{c[3]}' for c in CASES], summary_export,
                                        {'scale': 2})
        self.assertIs(p._pools['preprocessing'][1], pool)
        self.assertSameResults(got, self.in_process(p))

    def test_dead_worker_raises_and_recovers(self):
        p = self.predictor()
        with self.assertRaisesRegex(RuntimeError, 'worker of the streaming predictor died'):
            p.predict_cases_streaming([CASES[0], (9, (20, 20, 20), (1., 1., 1.), 'die'), CASES[1]], synthetic_load)
        self.assertNotIn('preprocessing', p._pools)
        got = p.predict_cases_streaming(CASES, synthetic_load, [f'out_{c[3]}' for c in CASES], summary_export,
                                        {'scale': 2})
        self.assertSameResults(got, self.in_process(p))

    def test_skip_failures(self):
        p = self.predictor()
        bad_load, bad_export = (9, (20, 20, 20), (1., 1., 1.), 'bad'), (8, (22, 20, 18), (1., 1., 1.), 'bad_export')
        cases = [CASES[0], bad_load, CASES[1], bad_export, CASES[2]]
        got = p.predict_cases_streaming(cases, synthetic_load, [f'out_{c[3]}' for c in cases], summary_export,
                                        {'scale': 2}, on_error='skip')
        self.assertIsInstance(got[1], CaseFailure)
        self.assertEqual((got[1].index, got[1].stage), (1, 'load'))
        self.assertIn('cannot load this case', got[1].traceback)
        self.assertIsInstance(got[3], CaseFailure)
        self.assertEqual((got[3].index, got[3].stage), (3, 'export'))
        self.assertSameResults([got[0], got[2], got[4]], self.in_process(p))
        failure = p.predict_single_case_streaming(bad_load, synthetic_load, None, summary_export, {'scale': 1},
                                                  on_error='skip')
        self.assertEqual((failure.index, failure.stage), (0, 'load'))
        with self.assertRaises(OSError):                       # default: raise
            p.predict_cases_streaming([CASES[0], bad_export], synthetic_load, None, summary_export, {'scale': 1})

    def test_progress_and_timings(self):
        p = self.predictor()
        for cases in (CASES, CASES[:1]):                           # pooled and in-process
            events = []
            p.predict_cases_streaming(cases, synthetic_load, [f'out_{c[3]}' for c in cases], summary_export,
                                      {'scale': 2}, progress_callback=lambda *e: events.append(e))
            for i in range(len(cases)):
                self.assertEqual([e for e, j, n in events if j == i], ['preprocessed', 'predicted', 'exported'])
            self.assertTrue(all(n == len(cases) for _, _, n in events))
            self.assertEqual(len(p.last_case_timings), len(cases))
            stages = {'load_and_preprocess', 'predict', 'export'}
            # in-process: loading overlaps getting the GPU ready, and both are recorded
            extra = {'gpu_ready', 'wait_for_preprocessing'} if len(cases) == 1 else set()
            for t in p.last_case_timings:
                self.assertEqual(set(t), stages | extra)
                self.assertTrue(all(t[k] > 0 for k in stages))
                self.assertTrue(all(t[k] >= 0 for k in extra))

    def test_gaussian_cache_kept_across_calls(self):
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        p = self.predictor()
        compute_gaussian.cache_clear()
        p.predict_cases_streaming(CASES, synthetic_load, [f'out_{c[3]}' for c in CASES], summary_export, {'scale': 1})
        first = compute_gaussian.cache_info()
        p.predict_cases_streaming(CASES, synthetic_load, [f'out_{c[3]}' for c in CASES], summary_export, {'scale': 1})
        second = compute_gaussian.cache_info()
        self.assertEqual((first.misses, second.misses), (1, 1))
        self.assertGreater(second.hits, first.hits)

    def test_more_resampling_threads_same_data(self):
        from nnunetv2.inference.streaming_pipeline import load_and_preprocess, with_resampling_threads
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        pm = PlansManager(_plans())
        cm = pm.get_configuration('3d_fullres')
        dataset_json = {'channel_names': {'0': 'noise'}, 'labels': {'background': 0, 'a': 1}, 'file_ending': '.nii.gz'}
        for case in CASES:
            ref, _ = load_and_preprocess(synthetic_load, case, pm, cm, dataset_json, cm.preprocessor_class(verbose=False))
            many, _ = load_and_preprocess(synthetic_load, case, pm, cm, dataset_json, cm.preprocessor_class(verbose=False),
                                          resampling_threads=16)
            self.assertTrue(np.array_equal(ref, many))
        self.assertEqual(with_resampling_threads(cm, 16).configuration['resampling_fn_data_kwargs']['num_threads'], 16)
        self.assertNotIn('num_threads', cm.configuration['resampling_fn_data_kwargs'])

    def test_parts(self):
        p = self.predictor()
        got = p.predict_cases_streaming(CASES, synthetic_load, [f'out_{c[3]}' for c in CASES], summary_export,
                                        {'scale': 2}, num_parts=2, part_id=0)
        self.assertSameResults(got, self.in_process(p, CASES[0::2]))

    def test_quiet(self):
        for quiet in (False, True):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                p = self.predictor(quiet=quiet)
                p.predict_cases_streaming(CASES[:2], synthetic_load, None, summary_export, {'scale': 1})
            self.assertEqual(buf.getvalue() == '', quiet, buf.getvalue()[:500])


class TestCompilePolicy(unittest.TestCase):
    def _resolve(self, requested, num_cases, env=None, available=True):
        environ = {k: v for k, v in os.environ.items() if k != 'nnUNet_compile'}
        if env is not None:
            environ['nnUNet_compile'] = env
        with mock.patch.dict(os.environ, environ, clear=True), \
                mock.patch.object(ps, 'compile_available', return_value=(available, 'test')):
            return ps.resolve_compile(requested, torch.device('cuda'), num_cases)

    def test_auto(self):
        for requested in ('auto', None):
            self.assertFalse(self._resolve(requested, 1))
            self.assertFalse(self._resolve(requested, ps.AUTO_COMPILE_MAX_CASES))
            self.assertTrue(self._resolve(requested, ps.AUTO_COMPILE_MAX_CASES + 1))
            self.assertFalse(self._resolve(requested, None))                # unknown length: no
            self.assertFalse(self._resolve(requested, 100, available=False))
            self.assertTrue(self._resolve(requested, 1, env='1'))           # the environment variable wins
            self.assertFalse(self._resolve(requested, 100, env='0'))
        self.assertTrue(self._resolve(True, 1))                              # explicit requests win
        self.assertFalse(self._resolve(False, 100, env='1'))

    def _prepared(self, compile_setting, want):
        p = nnUNetPredictorStreaming(device=CPU, compile=compile_setting, feature_ring=False, quiet=True)
        net = torch.nn.Conv3d(1, 2, 1)
        p.network = torch.compile(net)                     # lazy: nothing is compiled until it is called
        p.list_of_parameters = [net.state_dict()]
        with mock.patch.object(ps, 'resolve_compile', return_value=want), \
                mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != 'nnUNet_compile'}, clear=True):
            p._prepare_standalone_inference(num_cases=1)
        return ps.is_optimized_module(p.network)

    def test_auto_keeps_a_compiled_network(self):
        self.assertTrue(self._prepared('auto', want=False))   # an earlier batch compiled it: the cost is paid
        self.assertFalse(self._prepared(False, want=False))   # an explicit off unwraps

    def test_set_compile(self):
        p = nnUNetPredictorStreaming(device=CPU, quiet=True)
        self.assertEqual(p.compile, 'auto')
        p.set_compile(True)
        self.assertIs(p.compile, True)
        p.set_compile(None)
        self.assertEqual(p.compile, 'auto')
        with self.assertRaises(AssertionError):
            p.set_compile('yes')


class TestNibabelInMemory(unittest.TestCase):
    """read_nibabel_images / seg_to_nibabel = read_images / write_seg without files (what toolkits hand over)."""
    def test_same_as_files(self):
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
        with tempfile.TemporaryDirectory() as tmp:
            affine = np.array([[0., 0., -1.5, 10.], [-0.8, 0., 0., 5.], [0., 1.2, 0., -3.], [0., 0., 0., 1.]])
            img = nib.Nifti1Image(_image(4, (20, 18, 16))[0].astype(np.float32), affine)
            fname = join(tmp, 'img.nii.gz')
            nib.save(img, fname)
            for rw in (NibabelIO(), NibabelIOWithReorient()):
                data_f, props_f = rw.read_images([fname])
                data_m, props_m = rw.read_nibabel_images([nib.load(fname)])
                np.testing.assert_array_equal(data_m, data_f)
                self.assertEqual(props_m['spacing'], props_f['spacing'])
                for k, v in props_f['nibabel_stuff'].items():
                    np.testing.assert_array_equal(props_m['nibabel_stuff'][k], v)
                seg = (data_f[0] > 0).astype(np.uint8)
                rw.write_seg(seg, join(tmp, 'seg.nii.gz'), props_f)
                on_disk = nib.load(join(tmp, 'seg.nii.gz'))
                in_mem = rw.seg_to_nibabel(seg, props_m)
                np.testing.assert_array_equal(np.asanyarray(in_mem.dataobj), np.asanyarray(on_disk.dataobj))
                np.testing.assert_allclose(in_mem.affine, on_disk.affine)
                np.testing.assert_allclose(in_mem.affine, affine)           # back in the input's orientation


class TestSimpleITKInMemory(unittest.TestCase):
    """read_sitk_images / seg_to_sitk = read_images / write_seg without files, also for the reorienting reader."""
    def test_same_as_files(self):
        import SimpleITK as sitk
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO, SimpleITKIOWithReorient
        with tempfile.TemporaryDirectory() as tmp:
            img = sitk.GetImageFromArray(_image(5, (16, 18, 20))[0].astype(np.float32))
            img.SetSpacing((0.8, 1.2, 2.5))
            img.SetOrigin((10., -5., 3.))
            img.SetDirection((1., 0., 0., 0., -1., 0., 0., 0., 1.))          # LAS-like
            fname = join(tmp, 'img.nii.gz')
            sitk.WriteImage(img, fname)
            for rw in (SimpleITKIO(), SimpleITKIOWithReorient()):
                data_f, props_f = rw.read_images([fname])
                data_m, props_m = rw.read_sitk_images([sitk.ReadImage(fname)])
                np.testing.assert_array_equal(data_m, data_f)
                self.assertEqual(props_m['spacing'], props_f['spacing'])
                self.assertEqual(props_m['sitk_stuff'], props_f['sitk_stuff'])
                seg = (data_f[0] > 0).astype(np.uint8)
                rw.write_seg(seg, join(tmp, 'seg.nii.gz'), props_f)
                on_disk = sitk.ReadImage(join(tmp, 'seg.nii.gz'))
                in_mem = rw.seg_to_sitk(seg, props_m)
                np.testing.assert_array_equal(sitk.GetArrayFromImage(in_mem), sitk.GetArrayFromImage(on_disk))
                for get in ('GetSpacing', 'GetOrigin', 'GetDirection'):
                    np.testing.assert_allclose(getattr(in_mem, get)(), getattr(on_disk, get)(), atol=1e-6)
                    np.testing.assert_allclose(getattr(in_mem, get)(), getattr(img, get)(), atol=1e-6)


class TestExpandableSegments(unittest.TestCase):
    def _run(self, environ, initialized):
        with mock.patch.dict(os.environ, environ, clear=True), \
                mock.patch.object(torch.cuda, 'is_initialized', return_value=initialized):
            changed = ps.maybe_enable_expandable_segments()
            return changed, os.environ.get('PYTORCH_CUDA_ALLOC_CONF')

    def test_rules(self):
        self.assertEqual(self._run({}, False), (True, 'expandable_segments:True'))
        self.assertEqual(self._run({}, True), (False, None))                          # too late
        self.assertEqual(self._run({'PYTORCH_CUDA_ALLOC_CONF': 'x'}, False), (False, 'x'))   # never overrides
        self.assertEqual(self._run({'PYTORCH_ALLOC_CONF': 'y'}, False), (False, None))


if __name__ == '__main__':
    unittest.main()
