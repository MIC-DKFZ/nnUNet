"""Regression coverage for issue #3076, with real cropping, normalization and resampling."""

from copy import deepcopy
from unittest.mock import Mock

import blosc2
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from scipy.special import softmax

from nnunetv2.experiment_planning.like_nnssl import new_spacing_from_mode
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
    export_prediction_from_logits,
    resample_and_save,
)
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


FORWARD = [2, 0, 1]
BACKWARD = [1, 2, 0]
NATIVE_SPACING = [0.7, 1.6, 5.0]
TRANSPOSED_SPACING = [5.0, 0.7, 1.6]
RAW_SHAPE = (6, 8, 10)
RAW_CROP = (slice(1, 5), slice(1, 7), slice(2, 9))
CROPPED_SHAPE = (7, 4, 6)
DATASET_JSON = {'channel_names': {'0': 'test'}, 'labels': {'background': 0, 'foreground': 1},
                'file_ending': '.nii.gz'}

# Explicit expected values catch axis swaps and accidental use of dataset-median spacing.
SPACING_CASES = [
    pytest.param([None, None, None], [5.0, 0.7, 1.6], (7, 4, 6), id='3d-native'),
    pytest.param([None, 1.4, 0.8], [5.0, 1.4, 0.8], (7, 2, 12), id='3d-partial'),
    pytest.param([2.5, 1.4, 0.8], [2.5, 1.4, 0.8], (14, 2, 12), id='3d-numeric'),
    pytest.param([None, None], [5.0, 0.7, 1.6], (7, 4, 6), id='2d-native'),
    pytest.param([None, 0.8], [5.0, 0.7, 0.8], (7, 4, 12), id='2d-partial'),
    pytest.param([1.4, 0.8], [5.0, 1.4, 0.8], (7, 2, 12), id='2d-numeric'),
]


@pytest.fixture(autouse=True)
def restore_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        # The unpatched export function can fail before restoring its thread count.
        torch.set_num_threads(previous)


def make_managers(spacing):
    configuration = {
        'architecture': {},
        'spacing': list(spacing),
        'patch_size': [4, 6] if len(spacing) == 2 else [4, 4, 6],
        'normalization_schemes': ['ZScoreNormalization'],
        'use_mask_for_norm': [False],
    }
    for kind, order in [('data', 3), ('seg', 1), ('probabilities', 1)]:
        configuration['resampling_fn_' + kind] = 'resample_data_or_seg_to_shape'
        configuration['resampling_fn_' + kind + '_kwargs'] = {
            'is_seg': kind == 'seg', 'order': order, 'order_z': 0, 'force_separate_z': None,
        }
    plans = PlansManager({
        'transpose_forward': FORWARD, 'transpose_backward': BACKWARD,
        'foreground_intensity_properties_per_channel': {'0': {}},
        'image_reader_writer': 'SimpleITKIO',
        'original_median_spacing_after_transp': [99.0, 98.0, 97.0],
    })
    return plans, ConfigurationManager(configuration)


def make_case(spacing=NATIVE_SPACING):
    data = np.zeros((1, *RAW_SHAPE), dtype=np.float32)
    data[(0, *RAW_CROP)] = 1 + np.arange(4 * 6 * 7, dtype=np.float32).reshape(4, 6, 7)
    seg = np.zeros_like(data, dtype=np.int8)
    seg[(0, *RAW_CROP)] = np.indices((4, 6, 7)).sum(axis=0) % 2
    properties = {
        'spacing': list(spacing),
        'sitk_stuff': {'spacing': tuple(spacing[::-1]), 'origin': (10.0, 20.0, 30.0),
                       'direction': (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
    }
    return data, seg, properties


def export_properties():
    properties = make_case()[2]
    properties.update({
        'shape_before_cropping': (10, 6, 8),
        'shape_after_cropping_and_before_resampling': CROPPED_SHAPE,
        'bbox_used_for_cropping': [[2, 9], [1, 5], [1, 7]],
    })
    return properties


def make_logits(shape):
    labels = (np.indices(shape).sum(axis=0) % 2).astype(np.float32)
    return np.stack((1 - labels, labels)) * 6 - 3


def spy_on_resampler(monkeypatch, configuration, kind):
    name = 'resampling_fn_' + kind
    spy = Mock(wraps=getattr(configuration, name))
    monkeypatch.setattr(ConfigurationManager, name, property(lambda self: spy))
    return spy


def assert_resampling_call(spy, shape, current_spacing, target_spacing):
    spy.assert_called_once()
    args = spy.call_args.args
    assert tuple(args[1]) == tuple(shape)
    np.testing.assert_allclose(args[2], current_spacing)
    np.testing.assert_allclose(args[3], target_spacing)
    assert all(value is not None for value in (*args[2], *args[3]))


@pytest.mark.parametrize('configured,resolved,processed_shape', SPACING_CASES)
@pytest.mark.parametrize('has_seg', [True, False], ids=['training', 'inference'])
def test_preprocessing_resolves_spacing(monkeypatch, configured, resolved, processed_shape, has_seg):
    plans, configuration = make_managers(configured)
    data, seg, properties = make_case()
    original_data, original_seg = data.copy(), seg.copy()
    data_spy = spy_on_resampler(monkeypatch, configuration, 'data')
    seg_spy = spy_on_resampler(monkeypatch, configuration, 'seg')

    result, result_seg, properties = DefaultPreprocessor(verbose=False).run_case_npy(
        data, seg if has_seg else None, properties, plans, configuration, DATASET_JSON,
    )

    assert result.shape == (1, *processed_shape)
    assert result_seg.shape == (1, *processed_shape)
    assert result.dtype == np.float32 and result_seg.dtype == np.int8
    assert properties['shape_before_cropping'] == (10, 6, 8)
    assert properties['shape_after_cropping_and_before_resampling'] == CROPPED_SHAPE
    assert properties['bbox_used_for_cropping'] == [[2, 9], [1, 5], [1, 7]]
    assert properties['spacing'] == NATIVE_SPACING
    assert configuration.spacing == configured
    assert_resampling_call(data_spy, processed_shape, TRANSPOSED_SPACING, resolved)
    assert_resampling_call(seg_spy, processed_shape, TRANSPOSED_SPACING, resolved)
    np.testing.assert_array_equal(data, original_data)
    np.testing.assert_array_equal(seg, original_seg)

    cropped = original_data[(slice(None), *RAW_CROP)].transpose([0, 3, 1, 2])
    normalized = ZScoreNormalization(use_mask_for_norm=False, intensityproperties={}).run(cropped[0])
    expected_data = resample_data_or_seg_to_shape(
        normalized[None], processed_shape, TRANSPOSED_SPACING, resolved,
        is_seg=False, order=3, order_z=0, force_separate_z=None,
    )
    np.testing.assert_allclose(result, expected_data, rtol=1e-5, atol=1e-6)
    if has_seg:
        cropped_seg = original_seg[(slice(None), *RAW_CROP)].transpose([0, 3, 1, 2])
        expected_seg = resample_data_or_seg_to_shape(
            cropped_seg, processed_shape, TRANSPOSED_SPACING, resolved,
            is_seg=True, order=1, order_z=0, force_separate_z=None,
        )
        np.testing.assert_array_equal(result_seg, expected_seg)
        assert properties['present_labels'] == [int(value) for value in np.unique(expected_seg)]
    else:
        assert 'present_labels' not in properties
        np.testing.assert_array_equal(result_seg, np.zeros_like(result_seg))


@pytest.mark.parametrize('configured,resolved,processed_shape', SPACING_CASES)
@pytest.mark.parametrize('return_probabilities', [False, True], ids=['labels', 'probabilities'])
def test_export_restores_shape_and_orientation(monkeypatch, configured, resolved, processed_shape,
                                              return_probabilities):
    plans, configuration = make_managers(configured)
    logits = make_logits(processed_shape)
    properties = export_properties()
    spy = spy_on_resampler(monkeypatch, configuration, 'probabilities')
    result = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, plans, configuration, plans.get_label_manager(DATASET_JSON), properties,
        return_probabilities=return_probabilities, num_threads_torch=1,
    )
    segmentation, probabilities = result if return_probabilities else (result, None)
    assert_resampling_call(spy, CROPPED_SHAPE, resolved, TRANSPOSED_SPACING)
    assert configuration.spacing == configured
    assert properties['spacing'] == NATIVE_SPACING

    reference = resample_data_or_seg_to_shape(
        logits, CROPPED_SHAPE, resolved, TRANSPOSED_SPACING,
        is_seg=False, order=1, order_z=0, force_separate_z=None,
    )
    expected_seg = np.zeros(RAW_SHAPE, dtype=np.uint8)
    expected_seg[RAW_CROP] = reference.argmax(0).transpose(BACKWARD)
    np.testing.assert_array_equal(segmentation, expected_seg)
    if return_probabilities:
        expected_probabilities = np.zeros((2, *RAW_SHAPE), dtype=np.float32)
        expected_probabilities[0] = 1
        expected_probabilities[(slice(None), *RAW_CROP)] = softmax(reference, axis=0).transpose([0, 2, 3, 1])
        np.testing.assert_allclose(probabilities, expected_probabilities, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('configured,resolved,processed_shape', SPACING_CASES)
@pytest.mark.parametrize('change_shape', [False, True], ids=['same-grid', 'next-stage-grid'])
def test_resample_and_save_resolves_both_spacing_vectors(monkeypatch, tmp_path, configured, resolved,
                                                       processed_shape, change_shape):
    plans, configuration = make_managers(configured)
    logits = make_logits(processed_shape)
    target_shape = (5, 3, 4) if change_shape else processed_shape
    spy = spy_on_resampler(monkeypatch, configuration, 'probabilities')
    output = str(tmp_path / 'next_stage')
    resample_and_save(logits, list(target_shape), output, plans, configuration,
                      export_properties(), DATASET_JSON, num_threads_torch=1)
    assert_resampling_call(spy, target_shape, resolved, resolved)
    assert configuration.spacing == configured

    # Read the real default Blosc2 output; this function stores the target grid without undoing cropping/transpose.
    stored = blosc2.open(output + '.b2nd', mode='r')[:]
    reference = resample_data_or_seg_to_shape(
        logits, target_shape, resolved, resolved,
        is_seg=False, order=1, order_z=0, force_separate_z=None,
    ).argmax(0).astype(np.uint8)
    assert stored.shape == target_shape and stored.dtype == np.uint8
    np.testing.assert_array_equal(stored, reference)


@pytest.mark.parametrize('save_probabilities', [False, True])
def test_export_prediction_writes_native_geometry(tmp_path, save_probabilities):
    plans, configuration = make_managers([None, None, None])
    data, seg, properties = make_case()
    _, preprocessed_seg, properties = DefaultPreprocessor(verbose=False).run_case_npy(
        data, seg, properties, plans, configuration, DATASET_JSON,
    )
    logits = np.stack((1 - preprocessed_seg[0], preprocessed_seg[0])).astype(np.float32) * 6 - 3
    output = str(tmp_path / 'prediction')
    export_prediction_from_logits(torch.from_numpy(logits), properties, configuration, plans, DATASET_JSON,
                                  output, save_probabilities=save_probabilities, num_threads_torch=1)
    image = sitk.ReadImage(output + '.nii.gz')
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), seg[0])
    np.testing.assert_allclose(image.GetSpacing(), NATIVE_SPACING[::-1])
    np.testing.assert_allclose(image.GetOrigin(), properties['sitk_stuff']['origin'])
    np.testing.assert_allclose(image.GetDirection(), properties['sitk_stuff']['direction'])
    assert configuration.spacing == [None, None, None]
    if save_probabilities:
        with np.load(output + '.npz') as saved:
            assert saved['probabilities'].shape == (2, *RAW_SHAPE)
            np.testing.assert_allclose(saved['probabilities'].sum(axis=0), 1, atol=1e-6)
        assert (tmp_path / 'prediction.pkl').is_file()


@pytest.mark.parametrize('configured', [[None, None, None], [None, None]], ids=['3d', '2d'])
def test_native_spacing_is_per_case_without_mutating_plans(monkeypatch, tmp_path, configured):
    plans, configuration = make_managers(configured)
    original_configuration = deepcopy(configuration.configuration)
    spy = spy_on_resampler(monkeypatch, configuration, 'probabilities')
    for index, (native, expected_transposed) in enumerate([
        ([0.7, 1.6, 5.0], [5.0, 0.7, 1.6]),
        ([1.2, 2.3, 9.0], [9.0, 1.2, 2.3]),
    ]):
        data, seg, properties = make_case(native)
        result, preprocessed_seg, properties = DefaultPreprocessor(verbose=False).run_case_npy(
            data, seg, properties, plans, configuration, DATASET_JSON,
        )
        assert result.shape[1:] == CROPPED_SHAPE
        logits = np.stack((1 - preprocessed_seg[0], preprocessed_seg[0])).astype(np.float32) * 6 - 3
        spy.reset_mock()
        restored = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, plans, configuration, plans.get_label_manager(DATASET_JSON), properties, num_threads_torch=1,
        )
        assert_resampling_call(spy, CROPPED_SHAPE, expected_transposed, expected_transposed)
        np.testing.assert_array_equal(restored, seg[0])
        spy.reset_mock()
        resample_and_save(logits, list(CROPPED_SHAPE), str(tmp_path / str(index)), plans, configuration,
                          properties, DATASET_JSON, num_threads_torch=1)
        assert_resampling_call(spy, CROPPED_SHAPE, expected_transposed, expected_transposed)
        assert configuration.configuration == original_configuration
        assert properties['spacing'] == native


def test_no_resample_planning_preserves_per_case_sentinel():
    assert new_spacing_from_mode('no_resample', (0.7, 1.6, 5.0), (2.0, 2.0, 2.0), None) == [None, None, None]
