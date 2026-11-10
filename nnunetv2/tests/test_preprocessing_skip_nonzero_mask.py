"""
Inference preprocessing without the nonzero mask (DefaultPreprocessor.run_case_npy(..., return_seg=False)): same data
and properties as with the mask whenever it is skipped, and the old path whenever something needs the mask.
"""
import unittest
from copy import deepcopy
from unittest import mock

import numpy as np
import torch

from nnunetv2.inference import data_iterators
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor, skip_seg_kwargs
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

INTENSITIES = {'mean': 10.0, 'std': 30.0, 'percentile_00_5': -50.0, 'percentile_99_5': 80.0}


def _managers(schemes, use_mask_for_norm, spacing=(2.0, 1.5, 1.5), transpose=(0, 1, 2)):
    configuration = {
        'architecture': {}, 'spacing': list(spacing), 'patch_size': [8, 8, 8],
        'normalization_schemes': list(schemes), 'use_mask_for_norm': list(use_mask_for_norm),
        **{f'resampling_fn_{k}': 'resample_data_or_seg_to_shape' for k in ('data', 'seg', 'probabilities')},
        **{f'resampling_fn_{k}_kwargs': {'is_seg': k == 'seg', 'order': 3 if k == 'data' else 1, 'order_z': 0,
                                         'force_separate_z': None} for k in ('data', 'seg', 'probabilities')},
    }
    plans = PlansManager({
        'transpose_forward': list(transpose), 'transpose_backward': list(np.argsort(transpose)),
        'foreground_intensity_properties_per_channel': {str(c): INTENSITIES for c in range(len(schemes))},
        'image_reader_writer': 'SimpleITKIO',
    })
    return plans, ConfigurationManager(configuration)


def _image(rng, n_channels, shape=(24, 30, 27), border=(2, 5, 0, 3, 4, 1)):
    """nonzero inside a box away from (some) borders, with zero holes inside and a second channel that differs"""
    data = np.zeros((n_channels, *shape), dtype=np.float32)
    box = tuple(slice(border[2 * i], shape[i] - border[2 * i + 1]) for i in range(3))
    for c in range(n_channels):
        values = rng.normal(20, 40, size=shape).astype(np.float32)
        values[rng.random(shape) < 0.05] = 0  # holes
        data[c][box] = values[box]
    if n_channels > 1:
        data[1, :, :, : shape[2] // 2] = 0
    return data


def _run(preprocessor, data, seg, plans, configuration, **kwargs):
    properties = {'spacing': [3.0, 1.0, 1.25]}
    return preprocessor.run_case_npy(data, seg, properties, plans, configuration, {}, **kwargs)


class TestSkipNonzeroMask(unittest.TestCase):
    def setUp(self):
        self._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self._threads)

    def assert_same(self, a, b):
        self.assertTrue(np.array_equal(a[0], b[0]))
        self.assertEqual(a[0].dtype, b[0].dtype)
        self.assertEqual(a[2], b[2])

    def test_skipped_mask_gives_same_data_and_properties(self):
        rng = np.random.default_rng(0)
        pp = DefaultPreprocessor(verbose=False)
        for schemes in (['CTNormalization'], ['ZScoreNormalization'], ['CTNormalization', 'ZScoreNormalization'],
                        ['NoNormalization', 'RescaleTo01Normalization']):
            for transpose in ((0, 1, 2), (2, 0, 1)):
                for border in ((2, 5, 0, 3, 4, 1), (0, 0, 0, 0, 0, 0), (7, 3, 6, 9, 5, 2)):
                    with self.subTest(schemes=schemes, transpose=transpose, border=border):
                        plans, configuration = _managers(schemes, [False] * len(schemes), transpose=transpose)
                        data = _image(rng, len(schemes), border=border)
                        old = _run(pp, data, None, plans, configuration)
                        with mock.patch('nnunetv2.preprocessing.preprocessors.default_preprocessor.crop_to_nonzero',
                                        side_effect=AssertionError('the mask must not be built')):
                            new = _run(pp, data, None, plans, configuration, return_seg=False)
                        self.assert_same(new, old)
                        self.assertIsNone(new[1])
                        self.assertIsNotNone(old[1])

    def test_all_zero_image(self):
        pp = DefaultPreprocessor(verbose=False)
        plans, configuration = _managers(['CTNormalization'], [False])
        data = np.zeros((1, 12, 10, 9), dtype=np.float32)
        self.assert_same(_run(pp, data, None, plans, configuration, return_seg=False),
                         _run(pp, data, None, plans, configuration))

    def test_mask_is_built_when_needed(self):
        rng = np.random.default_rng(1)
        pp = DefaultPreprocessor(verbose=False)
        data = _image(rng, 2)

        # one channel normalizes within the mask: it is built (and differs from its box), but not returned
        plans, configuration = _managers(['CTNormalization', 'ZScoreNormalization'], [False, True])
        old = _run(pp, data, None, plans, configuration)
        with mock.patch.object(DefaultPreprocessor, '_normalize', autospec=True,
                               side_effect=DefaultPreprocessor._normalize) as normalize:
            new = _run(pp, data, None, plans, configuration, return_seg=False)
        self.assert_same(new, old)
        self.assertIsNone(new[1])
        self.assertTrue(np.any(normalize.call_args.args[2] == -1))
        self.assertTrue(np.any(old[1] == -1))

        # a custom normalization might use the mask regardless of use_mask_for_norm, unless it declares otherwise
        # itself (not inherited)
        class CustomNormalization(ZScoreNormalization):
            def run(self, image, seg=None):
                assert seg is not None
                return super().run(image, seg)

        class DeclaredNormalization(ZScoreNormalization):
            uses_seg_only_if_use_mask_for_norm = True

            def run(self, image, seg=None):
                return super().run(image, seg)

        class InheritedNormalization(ZScoreNormalization):
            pass

        plans, configuration = _managers(['ZScoreNormalization'], [False])
        expected = _run(pp, data[:1], None, plans, configuration)
        for normalizer, mask_built in ((CustomNormalization, True), (DeclaredNormalization, False),
                                       (InheritedNormalization, True)):
            with self.subTest(normalizer=normalizer.__name__), \
                    mock.patch.object(DefaultPreprocessor, '_get_normalizer_class', return_value=normalizer), \
                    mock.patch('nnunetv2.preprocessing.preprocessors.default_preprocessor.crop_to_nonzero',
                               wraps=crop_to_nonzero) as crop:
                new = _run(pp, data[:1], None, plans, configuration, return_seg=False)
                self.assertEqual(crop.called, mask_built)
                self.assertIsNone(new[1])
                self.assert_same(new, expected)

        # a preprocessor with its own _normalize might use the mask in its own way
        class OwnNormalize(DefaultPreprocessor):
            def _normalize(self, data, seg, configuration_manager, foreground_intensity_properties_per_channel):
                assert seg is not None
                return super()._normalize(data, seg, configuration_manager, foreground_intensity_properties_per_channel)

        new = _run(OwnNormalize(verbose=False), data[:1], None, plans, configuration, return_seg=False)
        self.assert_same(new, _run(pp, data[:1], None, plans, configuration))

        # a segmentation is preprocessed along (training, or the previous stage of a cascade)
        seg = rng.integers(0, 3, size=(1, *data.shape[1:])).astype(np.int8)
        old = _run(pp, data[:1], seg, plans, configuration)
        new = _run(pp, data[:1], seg, plans, configuration, return_seg=False)
        self.assert_same(new, old)
        self.assertTrue(np.array_equal(new[1], old[1]))

    def test_custom_preprocessor_with_old_signature(self):
        class OldPreprocessor(DefaultPreprocessor):
            def run_case_npy(self, data, seg, properties, plans_manager, configuration_manager, dataset_json):
                return super().run_case_npy(data, seg, properties, plans_manager, configuration_manager, dataset_json)

        pp = OldPreprocessor(verbose=False)
        self.assertEqual(skip_seg_kwargs(pp.run_case_npy), {})
        self.assertEqual(skip_seg_kwargs(pp.run_case), {'return_seg': False})
        self.assertEqual(skip_seg_kwargs(DefaultPreprocessor(verbose=False).run_case_npy), {'return_seg': False})

        # run_case only forwards return_seg if run_case_npy takes it
        plans, configuration = _managers(['CTNormalization'], [False])
        data = _image(np.random.default_rng(2), 1)
        properties = {'spacing': [3.0, 1.0, 1.25]}

        class FakeReader:
            def read_images(self, files):
                return data, deepcopy(properties)

        with mock.patch.object(PlansManager, 'image_reader_writer_class', new=property(lambda self: FakeReader)):
            out = pp.run_case(['a.nii.gz'], None, plans, configuration, {}, return_seg=False)
        self.assert_same(out, _run(DefaultPreprocessor(verbose=False), data, None, plans, configuration))
        self.assertIsNotNone(out[1])

    def test_predictor_input_unchanged(self):
        # PreprocessAdapterFromNpy (predict_single_npy_array) opts in; the network input must not change
        rng = np.random.default_rng(3)
        plans, configuration = _managers(['CTNormalization', 'ZScoreNormalization'], [False, False])
        images = [_image(rng, 2), _image(rng, 2, border=(0, 0, 3, 3, 0, 0))]
        properties = [{'spacing': [3.0, 1.0, 1.25]} for _ in images]
        dataset_json = {'labels': {'background': 0, 'a': 1}, 'channel_names': {'0': 'CT', '1': 'MR'}}

        with mock.patch.object(ConfigurationManager, 'preprocessor_class', DefaultPreprocessor):
            def batches(skip):
                with mock.patch.object(data_iterators, 'skip_seg_kwargs',
                                       side_effect=skip_seg_kwargs if skip else (lambda method: {})):
                    adapter = data_iterators.PreprocessAdapterFromNpy(images, None, deepcopy(properties), None, plans,
                                                                      dataset_json, configuration)
                    return [adapter.generate_train_batch() for _ in images]

            for new, old in zip(batches(True), batches(False)):
                self.assertTrue(torch.equal(new['data'], old['data']))
                self.assertEqual(new['data_properties'], old['data_properties'])


if __name__ == '__main__':
    unittest.main()
