"""
nnUNetPredictorStreaming on region-based (sigmoid) models, against nnU-Net's default predictor (nnUNetPredictor): the
same segmentation up to floating-point rounding at region boundaries.

The default predictor resamples the full logit volume, applies the sigmoid and paints every region whose probability is
> 0.5, in regions_class_order (LabelManager.convert_probabilities_to_segmentation). Streaming thresholds the resampled
logits at 0, slab by slab. Model folders are written by hand (plans.json, dataset.json with regions_class_order, a
.safetensors checkpoint of a small random PlainConvUNet whose output-layer biases are set so that each region covers
20-60 % of an image). Label definitions: nested regions; overlapping regions that are not nested, painted in an order
that is not sorted; regions with an ignore label. Models: 3d (logit ring, feature ring, the default plans' separate-z
resampling, a 2-fold ensemble with test-time mirroring) and 2d. Cases of different shapes and spacings, so both
resampling steps run, one of them anisotropic. CPU always; CUDA (with the fused kernels where Triton is available) when
present. The comparison must be able to fail: an argmax over the region logits disagrees with the default.
"""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming
from nnunetv2.utilities.checkpoints import save_checkpoint_file
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

# (labels, regions_class_order)
NESTED = ({'background': 0, 'whole': [1, 2, 3], 'core': [2, 3], 'enhancing': 3}, [1, 2, 3])
# 6 regions, more than the 4 decoder features: the feature ring applies
OVERLAPPING = ({'background': 0, 'all': [1, 2, 3, 4, 5, 6], 'a': [2, 3], 'b': [3, 4], 'c': [5, 6], 'd': 4, 'e': 6},
               [1, 2, 3, 5, 4, 6])
WITH_IGNORE = ({'background': 0, 'outer': [1, 2], 'inner': 1, 'ignore': 3}, [2, 1])

# (seed, shape, spacing); the model spacing is 1 mm. The last case is anisotropic: separate-z under the default plans
CASES = [(1, (36, 30, 33), (1.3, 0.8, 1.0)),
         (3, (41, 33, 29), (0.9, 1.2, 1.5)),
         (4, (9, 44, 38), (4.0, 0.9, 0.9))]


def _image(seed, shape):
    g = torch.Generator().manual_seed(seed)
    low = torch.randn((1, 1, *[max(2, s // 5) for s in shape]), generator=g)
    return F.interpolate(low, size=shape, mode='trilinear', align_corners=False)[0].numpy()


def _plans(dim, sepz):
    conv, norm = ('torch.nn.modules.conv.Conv3d', 'torch.nn.modules.instancenorm.InstanceNorm3d') if dim == 3 else \
        ('torch.nn.modules.conv.Conv2d', 'torch.nn.modules.instancenorm.InstanceNorm2d')
    arch = {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
            'arch_kwargs': {'n_stages': 3, 'features_per_stage': [4, 8, 16], 'conv_op': conv,
                            'kernel_sizes': [[3] * dim] * 3, 'strides': [[1] * dim, [2] * dim, [2] * dim],
                            'n_conv_per_stage': [2, 2, 2], 'n_conv_per_stage_decoder': [1, 1], 'conv_bias': True,
                            'norm_op': norm, 'norm_op_kwargs': {'eps': 1e-5, 'affine': True}, 'dropout_op': None,
                            'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU',
                            'nonlin_kwargs': {'inplace': True}},
            '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}
    if sepz:        # the default plans: scipy, order 1 in-plane, order 0 along a low-resolution axis
        fn = 'resample_data_or_seg_to_shape'
        rs = {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}
        rs_data, rs_seg = dict(rs, order=3), dict(rs, is_seg=True)
    else:
        fn = 'resample_torch_fornnunet'
        rs = rs_data = {'is_seg': False, 'force_separate_z': False, 'memefficient_seg_resampling': True}
        rs_seg = dict(rs, is_seg=True)
    cfg = {'data_identifier': f'test_{dim}d', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 2,
           'patch_size': [16] * dim, 'median_image_size_in_voxels': [40] * dim, 'spacing': [1.] * dim,
           'normalization_schemes': ['ZScoreNormalization'], 'use_mask_for_norm': [False],
           'resampling_fn_data': fn, 'resampling_fn_seg': fn, 'resampling_fn_probabilities': fn,
           'resampling_fn_data_kwargs': rs_data, 'resampling_fn_seg_kwargs': rs_seg,
           'resampling_fn_probabilities_kwargs': rs, 'architecture': arch, 'batch_dice': False}
    configuration = '3d_fullres' if dim == 3 else '2d'
    return configuration, {
        'dataset_name': 'Dataset999_StreamingRegionsTest', 'plans_name': 'nnUNetPlans',
        'original_median_spacing_after_transp': [1., 1., 1.], 'original_median_shape_after_transp': [40, 40, 40],
        'image_reader_writer': 'NibabelIOWithReorient', 'transpose_forward': [0, 1, 2], 'transpose_backward': [0, 1, 2],
        'configurations': {configuration: cfg}, 'experiment_planner_used': 'test', 'label_manager': 'LabelManager',
        'foreground_intensity_properties_per_channel': {'0': {'max': 3., 'mean': 0., 'median': 0., 'min': -3.,
                                                              'percentile_00_5': -2.5, 'percentile_99_5': 2.5,
                                                              'std': 1.}}}


def write_region_model(folder, label_def, dim=3, folds=1, sepz=False, tta=False):
    labels, regions_class_order = label_def
    configuration, plans = _plans(dim, sepz)
    arch = plans['configurations'][configuration]['architecture']
    num_heads = len(regions_class_order)
    maybe_mkdir_p(folder)
    save_json(plans, join(folder, 'plans.json'))
    save_json({'channel_names': {'0': 'noise'}, 'labels': labels, 'regions_class_order': regions_class_order,
               'numTraining': 1, 'file_ending': '.nii.gz'}, join(folder, 'dataset.json'))
    for fold in range(folds):
        torch.manual_seed(100 + fold)
        net = get_network_from_plans(arch['network_class_name'], arch['arch_kwargs'], arch['_kw_requires_import'], 1,
                                     num_heads, allow_init=True, deep_supervision=False).eval()
        with torch.no_grad():
            # region k is positive on a fraction q_k of a probe image (shrinking, as nested regions do)
            head = net.decoder.seg_layers[-1]
            head.weight.mul_(6)
            head.bias.zero_()
            probe = torch.from_numpy(_image(1000 + fold, (32,) * 3 if dim == 3 else (1, 32, 32)))
            probe = (probe - probe.mean()) / probe.std()
            out = net(probe[None] if dim == 3 else probe)[0]
            for k, q in enumerate(np.linspace(0.6, 0.2, num_heads)):
                head.bias[k] = -torch.quantile(out[k].flatten(), 1 - float(q))
        maybe_mkdir_p(join(folder, f'fold_{fold}'))
        save_checkpoint_file({'network_weights': net.state_dict(), 'trainer_name': 'nnUNetTrainer',
                              'init_args': {'configuration': configuration, 'fold': fold},
                              'inference_allowed_mirroring_axes': tuple(range(dim)) if tta else None},
                             join(folder, f'fold_{fold}', 'checkpoint_final.safetensors'))


class TestRegionsCPU(unittest.TestCase):
    DEV = torch.device('cpu')
    MIN_AGREEMENT = 0.999                # per case; both predictors sum the logits in fp16
    MIN_AGREEMENT_FEATURE_RING = 0.995   # the fp16 ring holds features: it rounds a different quantity

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._models = {}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        env = mock.patch.dict(os.environ, {'nnUNet_compile': 'false'})       # both predictors eager
        env.start()
        self.addCleanup(env.stop)
        cudnn = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        self.addCleanup(lambda: setattr(torch.backends.cudnn, 'benchmark', cudnn[0]))
        self.addCleanup(lambda: setattr(torch.backends.cudnn, 'deterministic', cudnn[1]))

    def model(self, name, label_def, **kwargs):
        if name not in self._models:
            self._models[name] = join(self._tmp.name, name)
            write_region_model(self._models[name], label_def, **kwargs)
        return self._models[name]

    def default(self, folder, folds=1, tta=False):
        p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
                            perform_everything_on_device=self.DEV.type == 'cuda', device=self.DEV, allow_tqdm=False)
        p.initialize_from_trained_model_folder(folder, tuple(range(folds)), 'checkpoint_final.safetensors')
        return p

    def streaming(self, folder, folds=1, tta=False, feature_ring=True):
        p = nnUNetPredictorStreaming(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta, device=self.DEV,
                                     allow_tqdm=False, compile=False, feature_ring=feature_ring, quiet=True)
        p.initialize_from_folder_base_trainer(folder, tuple(range(folds)))
        self.addCleanup(p.close)
        return p

    def assertMatchesDefault(self, default, streaming, label_def, cases=CASES, min_agreement=None):
        self.assertTrue(default.label_manager.has_regions and streaming.label_manager.has_regions)
        for seed, shape, spacing in cases:
            with self.subTest(case=seed):
                image, props = _image(seed, shape), {'spacing': list(spacing)}
                ref = default.predict_single_npy_array(image, dict(props))
                got = streaming.predict_single_npy_array_streaming(image, dict(props))
                self.assertEqual(got.shape, ref.shape)
                self.assertGreater(len(np.unique(ref)), 2)                       # regions overlap, not a constant
                self.assertLessEqual(set(np.unique(got).tolist()), {0, *label_def[1]})
                self.assertGreaterEqual(float(np.mean(got == ref)), min_agreement or self.MIN_AGREEMENT)

    def test_nested_regions(self):
        folder = self.model('nested', NESTED)
        p = self.streaming(folder)
        self.assertMatchesDefault(self.default(folder), p, NESTED)
        self.assertIn('logit ring is smaller', p.feature_ring_status)            # 3 regions < 4 features

    def test_overlapping_regions_feature_and_logit_ring(self):
        folder = self.model('overlapping', OVERLAPPING)
        default = self.default(folder)
        p = self.streaming(folder)
        self.assertMatchesDefault(default, p, OVERLAPPING, min_agreement=self.MIN_AGREEMENT_FEATURE_RING)
        self.assertTrue(p.feature_ring_status.startswith('on'), p.feature_ring_status)
        self.assertMatchesDefault(default, self.streaming(folder, feature_ring=False), OVERLAPPING)

    def test_default_plans_separate_z(self):
        folder = self.model('nested_sepz', NESTED, sepz=True)
        p = self.streaming(folder)
        decisions = []
        separate_z_for_case = p.separate_z_for_case

        def record(properties):
            decisions.append(separate_z_for_case(properties))
            return decisions[-1]
        p.separate_z_for_case = record
        self.assertMatchesDefault(self.default(folder), p, NESTED)
        self.assertIn((0, 'scipy'), decisions)                                   # the 4 mm case took separate-z

    def test_ignore_label(self):
        folder = self.model('ignore', WITH_IGNORE)
        p = self.streaming(folder)
        self.assertEqual(p.label_manager.ignore_label, 3)
        self.assertMatchesDefault(self.default(folder), p, WITH_IGNORE)          # never predicts the ignore label

    def test_ensemble_with_mirroring(self):
        folder = self.model('overlapping_2folds_tta', OVERLAPPING, folds=2, tta=True)
        self.assertMatchesDefault(self.default(folder, 2, True), self.streaming(folder, 2, True), OVERLAPPING,
                                  cases=CASES[:2])

    def test_2d(self):
        for name, label_def, folds, tta in (('nested_2d', NESTED, 1, False),
                                            ('overlapping_2d_2folds_tta', OVERLAPPING, 2, True)):
            with self.subTest(model=name):
                folder = self.model(name, label_def, dim=2, folds=folds, tta=tta)
                self.assertMatchesDefault(self.default(folder, folds, tta), self.streaming(folder, folds, tta),
                                          label_def)

    def test_the_comparison_can_fail(self):
        # the softmax rule (argmax over the region logits) must not pass for the region rule
        folder = self.model('overlapping', OVERLAPPING)
        p = self.streaming(folder, feature_ring=False)
        p._use_triton_kernels = False
        regions_class_order = torch.tensor(OVERLAPPING[1])
        p._reduce_to_seg = lambda logits, dtype: regions_class_order.to(logits.device)[logits.argmax(0)].to(dtype)
        image, props = _image(*CASES[0][:2]), {'spacing': list(CASES[0][2])}
        ref = self.default(folder).predict_single_npy_array(image, dict(props))
        got = p.predict_single_npy_array_streaming(image, dict(props))
        self.assertLess(float(np.mean(got == ref)), 0.9)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA')
class TestRegionsCUDA(TestRegionsCPU):
    """The same on CUDA: both networks in fp16 (autocast), the fused kernels where Triton is available."""
    DEV = torch.device('cuda')
    MIN_AGREEMENT = 0.998


if __name__ == '__main__':
    unittest.main()
