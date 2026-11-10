"""
nnUNetPredictor keeps what does not change between cases: a single model's weights stay loaded, and compute_gaussian's
cache survives the end of a prediction call. Both must leave the predictions unchanged.
"""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.checkpoints import save_checkpoint_file
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

PATCH_SIZE = [16, 16, 16]
ARCHITECTURE = {
    'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
    'arch_kwargs': {
        'n_stages': 2, 'features_per_stage': [4, 8], 'conv_op': 'torch.nn.modules.conv.Conv3d',
        'kernel_sizes': [[3, 3, 3]] * 2, 'strides': [[1, 1, 1], [2, 2, 2]], 'n_conv_per_stage': [1, 1],
        'n_conv_per_stage_decoder': [1], 'conv_bias': True,
        'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
        'dropout_op': None, 'dropout_op_kwargs': None,
        'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True},
    },
    '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin'],
}
DATASET_JSON = {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'a': 1, 'b': 2}, 'file_ending': '.nii.gz'}
PLANS = {
    'dataset_name': 'Dataset999_Test', 'plans_name': 'nnUNetPlans', 'image_reader_writer': 'SimpleITKIO',
    'transpose_forward': [0, 1, 2], 'transpose_backward': [0, 1, 2],
    'original_median_spacing_after_transp': [1.0, 1.0, 1.0], 'original_median_shape_after_transp': [32, 32, 32],
    'foreground_intensity_properties_per_channel': {'0': {}},
    'configurations': {'3d_fullres': {
        'patch_size': PATCH_SIZE, 'spacing': [1.0, 1.0, 1.0], 'architecture': ARCHITECTURE,
        'normalization_schemes': ['ZScoreNormalization'], 'use_mask_for_norm': [False],
        **{f'resampling_fn_{k}': 'resample_data_or_seg_to_shape' for k in ('data', 'seg', 'probabilities')},
        **{f'resampling_fn_{k}_kwargs': {'is_seg': k == 'seg', 'order': 1, 'order_z': 0, 'force_separate_z': None}
           for k in ('data', 'seg', 'probabilities')},
    }},
}


def _network():
    return get_network_from_plans(ARCHITECTURE['network_class_name'], ARCHITECTURE['arch_kwargs'],
                                  ARCHITECTURE['_kw_requires_import'], 1, 3, allow_init=True, deep_supervision=False)


def _fold_weights(n_folds):
    weights = []
    for f in range(n_folds):
        torch.manual_seed(f)
        weights.append({k: v.clone() for k, v in _network().state_dict().items()})
    return weights


def _write_model_folder(folder, weights):
    save_json(PLANS, join(folder, 'plans.json'))
    save_json(DATASET_JSON, join(folder, 'dataset.json'))
    for f, w in enumerate(weights):
        maybe_mkdir_p(join(folder, f'fold_{f}'))
        save_checkpoint_file({'network_weights': w, 'trainer_name': 'nnUNetTrainer',
                              'init_args': {'configuration': '3d_fullres'},
                              'inference_allowed_mirroring_axes': (0, 1, 2)},
                             join(folder, f'fold_{f}', 'checkpoint_final.safetensors'))


def _predictor_from_folder(weights):
    predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False, allow_tqdm=False)
    with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {'nnUNet_compile': 'f'}):
        _write_model_folder(folder, weights)
        predictor.initialize_from_trained_model_folder(folder, use_folds=tuple(range(len(weights))))
    return predictor


def _expected_logits(predictor, weights, data):
    """what predict_logits_from_preprocessed_data computed when it loaded every state dict for every case"""
    expected = None
    for w in weights:
        predictor.network.load_state_dict(w)
        logits = predictor.predict_sliding_window_return_logits(data).to('cpu')
        expected = logits if expected is None else expected + logits
    if len(weights) > 1:
        expected /= len(weights)
    return expected


def _cases(n):
    rng = np.random.default_rng(0)
    return [torch.from_numpy(rng.normal(size=(1, 20 + 3 * i, 18, 17)).astype(np.float32)) for i in range(n)]


@mock.patch.dict(os.environ, {'nnUNet_compile': 'f'})
class TestWeightsStayLoaded(unittest.TestCase):
    def setUp(self):
        self._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self._threads)

    def _predict_counting_loads(self, predictor, data):
        module = getattr(predictor.network, '_orig_mod', predictor.network)
        with mock.patch.object(module, 'load_state_dict', wraps=module.load_state_dict) as spy:
            logits = predictor.predict_logits_from_preprocessed_data(data)
        return logits, spy.call_count

    def test_single_fold_is_not_reloaded(self):
        weights = _fold_weights(1)
        predictor = _predictor_from_folder(weights)
        reference = _predictor_from_folder(weights)
        for data in _cases(3):
            logits, n_loads = self._predict_counting_loads(predictor, data)
            self.assertEqual(n_loads, 0)
            self.assertTrue(torch.equal(logits, _expected_logits(reference, weights, data)))

    def test_fold_ensemble_loads_every_fold(self):
        weights = _fold_weights(3)
        predictor = _predictor_from_folder(weights)
        reference = _predictor_from_folder(weights)
        for i, data in enumerate(_cases(2)):
            logits, n_loads = self._predict_counting_loads(predictor, data)
            # fold 0 is still loaded from initialization for the first case only
            self.assertEqual(n_loads, 2 if i == 0 else 3)
            self.assertTrue(torch.equal(logits, _expected_logits(reference, weights, data)))

    def test_manual_initialization_loads_once(self):
        weights = _fold_weights(2)
        network = _network()  # random weights, neither of the two
        reference = _predictor_from_folder(weights[:1])
        predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False, allow_tqdm=False)
        predictor.manual_initialization(network, reference.plans_manager, reference.configuration_manager,
                                        weights[:1], DATASET_JSON, 'nnUNetTrainer', (0, 1, 2))
        for i, data in enumerate(_cases(3)):
            logits, n_loads = self._predict_counting_loads(predictor, data)
            self.assertEqual(n_loads, 1 if i == 0 else 0)
            self.assertTrue(torch.equal(logits, _expected_logits(reference, weights[:1], data)))

    def test_replaced_network_or_parameters_are_loaded(self):
        weights = _fold_weights(2)
        predictor = _predictor_from_folder(weights[:1])
        reference = _predictor_from_folder(weights[:1])
        data = _cases(1)[0]

        predictor.network = _network()
        logits, n_loads = self._predict_counting_loads(predictor, data)
        self.assertEqual(n_loads, 1)
        self.assertTrue(torch.equal(logits, _expected_logits(reference, weights[:1], data)))

        predictor.list_of_parameters = [weights[1]]
        logits, n_loads = self._predict_counting_loads(predictor, data)
        self.assertEqual(n_loads, 1)
        self.assertTrue(torch.equal(logits, _expected_logits(reference, weights[1:], data)))


@mock.patch.dict(os.environ, {'nnUNet_compile': 'f'})
class TestGaussianCacheIsKept(unittest.TestCase):
    def test_second_call_hits_the_cache(self):
        weights = _fold_weights(1)
        predictor = _predictor_from_folder(weights)
        properties = {'spacing': [1.0, 1.0, 1.0], 'shape_before_cropping': (20, 18, 17),
                      'shape_after_cropping_and_before_resampling': (20, 18, 17),
                      'bbox_used_for_cropping': [[0, 20], [0, 18], [0, 17]]}
        data = _cases(1)[0]
        compute_gaussian.cache_clear()
        segmentations = []
        for _ in range(2):
            segmentations += predictor.predict_from_data_iterator(
                iter([{'data': data, 'ofile': None, 'data_properties': properties}]),
                num_processes_segmentation_export=1)
        info = compute_gaussian.cache_info()
        self.assertEqual(info.misses, 1)
        self.assertGreaterEqual(info.hits, 1)
        self.assertTrue(np.array_equal(segmentations[0], segmentations[1]))


if __name__ == '__main__':
    unittest.main()
