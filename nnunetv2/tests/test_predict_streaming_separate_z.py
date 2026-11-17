"""
Separate-z support of nnUNetPredictorStreaming: the per-case decision, the source index rules, and the
streamed resample + argmax against the two reference resamplers (resample_torch_fornnunet,
resample_data_or_seg_to_shape) on logits that take their separate-z path.

What these cannot show is the full pipeline on a trained model (sliding window, ring buffer, cropping); that
was checked against nnUNetv2_predict on trained models with separate-z plans.
"""
import contextlib
import io
import unittest
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch

from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming, nearest_source_index, separate_z_rule
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg, \
    resample_data_or_seg_to_shape
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet, resample_torch_simple

TORCH_SEPZ = partial(resample_torch_fornnunet, is_seg=False, force_separate_z=None, memefficient_seg_resampling=True)
SCIPY_SEPZ = partial(resample_data_or_seg_to_shape, is_seg=False, order=1, order_z=0, force_separate_z=None)


def _predictor(resampling_fn=TORCH_SEPZ, model_spacing=(1., 1., 1.), num_classes=5, force_trilinear=False):
    p = nnUNetPredictorStreaming(device=torch.device('cpu'), force_trilinear=force_trilinear)
    p.label_manager = SimpleNamespace(has_regions=False, regions_class_order=None,
                                      foreground_labels=list(range(1, num_classes)))
    p.configuration_manager = SimpleNamespace(resampling_fn_probabilities=resampling_fn, spacing=list(model_spacing),
                                              configuration={})
    p.plans_manager = SimpleNamespace(transpose_forward=[0, 1, 2])
    return p


def _logits(shape, num_classes=5, seed=0):
    # smooth logits (upsampled noise) so the argmax has realistic regions rather than per-voxel noise
    g = torch.Generator().manual_seed(seed)
    low = torch.randn((1, num_classes, *[max(2, s // 4) for s in shape]), generator=g)
    return torch.nn.functional.interpolate(low, size=shape, mode='trilinear', align_corners=False)[0]


class TestSeparateZRule(unittest.TestCase):
    def test_rules(self):
        cases = [
            (partial(resample_torch_fornnunet, is_seg=False, force_separate_z=False), (True, 'nearest-exact')),
            (TORCH_SEPZ, (True, 'nearest-exact')),
            (partial(resample_torch_fornnunet, is_seg=False, aniso_axis_mode='nearest'), (True, 'nearest')),
            (partial(resample_torch_fornnunet, is_seg=False, aniso_axis_mode='linear'), (True, None)),
            (partial(resample_torch_fornnunet, is_seg=False, mode='nearest'), (False, None)),
            (SCIPY_SEPZ, (True, 'scipy')),
            (partial(resample_data_or_seg_to_shape, is_seg=False, order=1, order_z=1), (True, None)),
            (partial(resample_data_or_seg_to_shape, is_seg=False, order=3, order_z=0), (False, None)),
            (partial(resample_torch_simple, is_seg=False), (False, None)),
        ]
        for fn, expected in cases:
            self.assertEqual(separate_z_rule(fn), expected, fn)


class TestNearestSourceIndex(unittest.TestCase):
    PAIRS = [(10, 10), (10, 25), (25, 10), (20, 10), (10, 20), (7, 3), (3, 7), (149, 124), (89, 148), (1, 5)]

    def test_scipy_rule_is_the_index_resample_data_or_seg_takes(self):
        for src, tgt in self.PAIRS:
            data = np.broadcast_to(np.arange(src, dtype=np.float32)[None, :, None, None], (1, src, 3, 3)).copy()
            out = resample_data_or_seg(data, (tgt, 3, 3), is_seg=False, axis=0, order=1, do_separate_z=True,
                                       order_z=0)
            np.testing.assert_array_equal(out[0, :, 1, 1].astype(np.int64),
                                          nearest_source_index(src, tgt, 'scipy').numpy(), err_msg=f'{src}->{tgt}')

    def test_torch_rule_is_the_index_resample_torch_fornnunet_takes(self):
        for src, tgt in self.PAIRS:
            data = torch.arange(src, dtype=torch.float32)[None, :, None, None].expand(1, src, 3, 3).contiguous()
            out = resample_torch_fornnunet(data, (tgt, 3, 3), (4., 1., 1.), (4., 1., 1.), is_seg=False,
                                           force_separate_z=True)
            np.testing.assert_array_equal(out[0, :, 1, 1].long().numpy(),
                                          nearest_source_index(src, tgt, 'nearest-exact').numpy(),
                                          err_msg=f'{src}->{tgt}')


class TestSeparateZForCase(unittest.TestCase):
    """The decision has to be the default export's: same function, same spacings, same kwargs."""

    def _props(self, spacing, shape=(10, 37, 37)):
        return {'spacing': list(spacing), 'shape_after_cropping_and_before_resampling': shape}

    def test_axes_the_plans_do_not_resample(self):
        """Plans spacing None (an axis the plans keep at the case's spacing, see test_no_resample.py): the case's own
        spacing stands in for it, as in export_prediction.convert_predicted_logits_to_segmentation_with_correct_shape,
        so the decision equals the one for the case's spacing written out."""
        case = self._props((4., .8, .8))
        for fn in (TORCH_SEPZ, SCIPY_SEPZ):
            for model_spacing, resolved in (((None, None, None), (4., .8, .8)), ((None, 1., 1.), (4., 1., 1.)),
                                            ((1., None, None), (1., .8, .8))):
                with self.subTest(fn=fn.func.__name__, model_spacing=model_spacing):
                    self.assertEqual(_predictor(fn, model_spacing=model_spacing).separate_z_for_case(case),
                                     _predictor(fn, model_spacing=resolved).separate_z_for_case(case))

    def test_notice_with_axes_the_plans_do_not_resample(self):
        p = _predictor(SCIPY_SEPZ, model_spacing=(None, 1., 1.))
        p.list_of_parameters = [{}]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            p.print_streaming_deviation_notice()
        self.assertIn('Resampling as configured in the plans', out.getvalue())

    def test_decision(self):
        aniso, iso = self._props((4., .8, .8)), self._props((1., 1., 1.))
        self.assertEqual(_predictor(TORCH_SEPZ).separate_z_for_case(aniso), (0, 'nearest-exact'))
        self.assertEqual(_predictor(SCIPY_SEPZ).separate_z_for_case(self._props((.8, .8, 4.))), (2, 'scipy'))
        self.assertIsNone(_predictor(TORCH_SEPZ).separate_z_for_case(iso))
        # torchres-style plans: force_separate_z=False never separates, whatever the case spacing
        no_sep = partial(resample_torch_fornnunet, is_seg=False, force_separate_z=False)
        self.assertIsNone(_predictor(no_sep).separate_z_for_case(aniso))
        self.assertIsNone(_predictor(TORCH_SEPZ, force_trilinear=True).separate_z_for_case(aniso))
        # an anisotropic model spacing decides first (its low-res axis), as in determine_do_sep_z_and_axis
        self.assertEqual(_predictor(TORCH_SEPZ, model_spacing=(1., 1., 5.)).separate_z_for_case(aniso),
                         (2, 'nearest-exact'))
        # transpose_forward: the properties' spacing is in the original axis order
        p = _predictor(TORCH_SEPZ)
        p.plans_manager.transpose_forward = [2, 0, 1]
        self.assertEqual(p.separate_z_for_case(self._props((.8, .8, 4.))), (0, 'nearest-exact'))


class TestStreamedResampleMatchesReference(unittest.TestCase):
    """resample_argmax_reference_streaming = the streamed per-slab resampler, fed finished logits."""
    MODEL_SPACING = (1., 1., 1.)

    def _check(self, case_spacing, source_shape, reference_fn, rule, exact_on_slab_axis):
        target = [int(round(s * m / c)) for s, m, c in zip(source_shape, self.MODEL_SPACING, case_spacing)]
        logits = _logits(source_shape)
        p = _predictor(reference_fn)
        sep = p.separate_z_for_case({'spacing': list(case_spacing),
                                     'shape_after_cropping_and_before_resampling': target})
        self.assertIsNotNone(sep)
        self.assertEqual(sep[1], rule)
        ref = reference_fn(logits.numpy() if reference_fn is SCIPY_SEPZ else logits, target,
                           list(self.MODEL_SPACING), list(case_spacing))
        ref = np.asarray(ref).argmax(0)
        for slab_axis in range(3):
            got = p.resample_argmax_reference_streaming(logits, target, slab_axis=slab_axis, separate_z=sep)
            self.assertEqual(tuple(got.shape), tuple(target))
            if exact_on_slab_axis and slab_axis == sep[0]:
                np.testing.assert_array_equal(got, ref, err_msg=f'slab axis {slab_axis}')
            else:
                self.assertGreater(np.mean(got == ref), 0.9999, f'slab axis {slab_axis}')
            trilinear = p.resample_argmax_reference_streaming(logits, target, slab_axis=slab_axis)
            self.assertLess(np.mean(trilinear == ref), np.mean(got == ref), 'the check must be able to fail')

    def test_torch_lowres_axis_0(self):
        self._check((4., .8, .8), (40, 30, 28), TORCH_SEPZ, 'nearest-exact', exact_on_slab_axis=True)

    def test_torch_lowres_axis_2(self):
        self._check((.7, .7, 3.), (30, 28, 50), TORCH_SEPZ, 'nearest-exact', exact_on_slab_axis=True)

    def test_scipy_lowres_axis_0(self):
        # skimage / map_coordinates compute in float64, streaming in float32: agreement, not identity
        self._check((4., .8, .8), (40, 30, 28), SCIPY_SEPZ, 'scipy', exact_on_slab_axis=False)

    def test_scipy_lowres_axis_1(self):
        self._check((.8, 5., .8), (30, 52, 28), SCIPY_SEPZ, 'scipy', exact_on_slab_axis=False)


if __name__ == '__main__':
    unittest.main()
