"""
nnUNetPredictorStreaming's per-case costs: weights stay loaded across cases, fold ensembles load from pinned host or
device-resident state dicts, patches are cut from the device-resident input, and the allocator cache is kept between
cases and emptied once per entry-point call. None of this may change the segmentation.

- the network is loaded once for a single model and again only when it, list_of_parameters or a fold's state dict is
  replaced, or after the logit method of nnUNetPredictor (which loads every fold) ran;
- several cases on one predictor give the same segmentations as a fresh predictor per case (CPU, exact);
- the cores never empty the allocator cache; _cache_emptied_after does, once, also when the call fails;
- 2d: the same, single model and fold ensemble; the ensemble's per-fold networks are built once and rebuilt when the
  state dicts are replaced;
- CUDA: a fold ensemble gives the same segmentation from pageable (the old path), pinned and device-resident state dicts
  (deterministic cuDNN, one process); a single model's and a 2d ensemble's state dicts are left as they are.
"""
import unittest
from functools import partial
from unittest import mock

import numpy as np
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming, unwrap_network
from nnunetv2.tests.test_predict_streaming_feature_ring import GEOMETRIES, _Base, _case, _net, _predictor, _state

CUDA = torch.cuda.is_available()


def _count_loads(predictor):
    """Patcher wrapping the network's load_state_dict; entering it gives the mock (call_count)."""
    target = unwrap_network(predictor.network)
    return mock.patch.object(target, 'load_state_dict', wraps=target.load_state_dict)


class TestWeightsCPU(_Base):
    DEV = torch.device('cpu')

    def test_single_model_loaded_once(self):
        p = _predictor(self.DEV)
        p._prepare_standalone_inference()
        with _count_loads(p) as loads:
            for src, tgt, *_ in GEOMETRIES[:2]:
                data, props = _case(src, tgt)
                p._predict_and_export_streaming(data, props)
                p._predict_and_export_streaming(data, props)
        self.assertEqual(loads.call_count, 1)

    def test_reload_when_replaced(self):
        # logit ring: with the feature ring the output layer is detached and the state dicts lose its entries
        p = _predictor(self.DEV, feature_ring=False)
        p._prepare_standalone_inference()
        data, props = _case(*GEOMETRIES[0][:2])
        with _count_loads(p) as loads:
            p._predict_and_export_streaming(data, props)
            self.assertEqual(loads.call_count, 1)
            # new list_of_parameters with another model's weights: loaded, and the output is that model's
            p.list_of_parameters = [_state(_net(seed=3))]
            got = p._predict_and_export_streaming(data, props)
            self.assertEqual(loads.call_count, 2)
            # one fold's state dict replaced in place in the list: loaded as well
            p.list_of_parameters[0] = _state(_net(seed=4))
            p._predict_and_export_streaming(data, props)
            self.assertEqual(loads.call_count, 3)
            # a new network (same architecture): its own load_state_dict is called, the old one's not again
            p.network = _net(seed=5)
            with _count_loads(p) as loads_new:
                p._predict_and_export_streaming(data, props)
            self.assertEqual((loads.call_count, loads_new.call_count), (3, 1))
        fresh = _predictor(self.DEV, feature_ring=False)
        fresh.list_of_parameters = [_state(_net(seed=3))]
        fresh._prepare_standalone_inference()
        np.testing.assert_array_equal(got, fresh._predict_and_export_streaming(data, props))

    def test_parent_logit_method_invalidates(self):
        p = _predictor(self.DEV, folds=2, feature_ring=False)
        p._prepare_standalone_inference()
        data, props = _case(*GEOMETRIES[0][:2])
        ref = p._predict_and_export_streaming(data, props)        # leaves fold 1 loaded

        def parent(self_, x):                                     # stands in for the parent: loads fold 0 last
            unwrap_network(self_.network).load_state_dict(self_.list_of_parameters[0])
        with mock.patch.object(nnUNetPredictor, 'predict_logits_from_preprocessed_data', parent):
            p.predict_logits_from_preprocessed_data(data)
        self.assertIsNone(p._loaded)
        np.testing.assert_array_equal(p._predict_and_export_streaming(data, props), ref)

    def test_many_cases_equal_fresh_predictor(self):
        cases = [_case(src, tgt, osp, seed=i) for i, (src, tgt, _, osp, *_) in enumerate(GEOMETRIES)]
        cases.append(cases[0])
        for folds in (1, 2):
            with self.subTest(folds=folds):
                p = _predictor(self.DEV, folds=folds)
                p._prepare_standalone_inference()
                for data, props in cases:
                    fresh = _predictor(self.DEV, folds=folds)
                    fresh._prepare_standalone_inference()
                    np.testing.assert_array_equal(p._predict_and_export_streaming(data, props),
                                                  fresh._predict_and_export_streaming(data, props))

    def test_cache_emptied_once_per_call(self):
        p = _predictor(self.DEV)
        p._prepare_standalone_inference()
        data, props = _case(*GEOMETRIES[0][:2])
        with mock.patch('nnunetv2.inference.predict_streaming.empty_cache') as ec:
            p._predict_and_export_streaming(data, props)
            p._predict_and_export_streaming(data, props)
            self.assertEqual(ec.call_count, 0)                    # the core keeps the cache across cases
            with self.assertRaises(RuntimeError):
                with p._cache_emptied_after():
                    p._predict_and_export_streaming(data, props)
                    raise RuntimeError('a case failed')
            self.assertEqual(ec.call_count, 1)

    def test_2d_many_cases_equal_fresh_predictor(self):
        cases = [_case((12, 36, 44), (12, 30, 52), seed=1), _case((9, 40, 33), (9, 57, 31), seed=2)]
        cases.append(cases[0])
        for folds in (1, 2):
            with self.subTest(folds=folds):
                p = _predictor(self.DEV, folds=folds, patch=(16, 16))
                p._prepare_standalone_inference()
                for data, props in cases:
                    fresh = _predictor(self.DEV, folds=folds, patch=(16, 16))
                    fresh._prepare_standalone_inference()
                    np.testing.assert_array_equal(p._predict_and_export_streaming(data, props),
                                                  fresh._predict_and_export_streaming(data, props))

    def test_2d_fold_networks_built_once(self):
        p = _predictor(self.DEV, folds=2, patch=(16, 16))
        p._prepare_standalone_inference()
        data, props = _case((12, 36, 44), (12, 30, 52))
        p._predict_and_export_streaming(data, props)
        nets = p._fold_networks_2d()
        p._predict_and_export_streaming(data, props)
        self.assertIs(p._fold_networks_2d(), nets)
        p.list_of_parameters = [_state(_net(seed=s, dim=2)) for s in (3, 4)]
        self.assertIsNot(p._fold_networks_2d(), nets)

    def test_cpu_keeps_list_of_parameters(self):
        p = _predictor(self.DEV, folds=2, feature_ring=False)
        params = p.list_of_parameters
        p._prepare_standalone_inference()
        self.assertIs(p.list_of_parameters, params)


@unittest.skipUnless(CUDA, 'CUDA')
class TestWeightsCUDA(_Base):
    DEV = torch.device('cuda')

    def test_ensemble_fold_weights_identical(self):
        data, props = _case(*GEOMETRIES[2][:2], original_spacing=GEOMETRIES[2][3])
        make = partial(_predictor, self.DEV, folds=3, model_spacing=GEOMETRIES[2][4], sepz=True)
        placed = {'pinned': torch.Tensor.is_pinned, 'device': lambda v: v.is_cuda}
        outs = {}
        for mode in ('pageable', 'pinned', 'device'):           # one process, fixed order
            p = make()
            p.fold_weights_on_device = mode == 'device'
            if mode == 'pageable':                              # the old path: list_of_parameters as loaded
                p._place_fold_weights = lambda: None
            p._prepare_standalone_inference()
            if mode != 'pageable':
                self.assertTrue(all(placed[mode](v) for sd in p.list_of_parameters for v in sd.values()))
            outs[mode] = [p._predict_and_export_streaming(data, props) for _ in range(2)]
        for mode in ('pinned', 'device'):
            for got, ref in zip(outs[mode], outs['pageable']):
                np.testing.assert_array_equal(got, ref)

    def test_single_model_state_dict_unchanged(self):
        p = _predictor(self.DEV, feature_ring=False)             # (the feature ring would replace the dicts itself)
        p.fold_weights_on_device = True
        params = p.list_of_parameters
        p._prepare_standalone_inference()
        self.assertIs(p.list_of_parameters, params)

    def test_2d_ensemble_state_dicts_unchanged(self):
        p = _predictor(self.DEV, folds=2, patch=(16, 16))         # 2d keeps one network per fold
        p.fold_weights_on_device = True
        params = p.list_of_parameters
        p._prepare_standalone_inference()
        self.assertIs(p.list_of_parameters, params)


if __name__ == '__main__':
    unittest.main()
