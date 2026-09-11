import os
import pickle
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.sampling_locations.extract_sampling_locations import (
    get_classes_or_regions_to_collect)
from nnunetv2.training.dataloading.foreground_locations import (
    FG_SAMPLING_PREFIX, ForegroundLocations, ForegroundLocationsWriter, LegacyForegroundLocations,
    get_foreground_locations, has_foreground_locations, ravel_coords)
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def _random_runs(rng, shape, class_keys, absent_probability=0.3, max_n=5000):
    """-> dict class key -> sorted linear indices, plus the same thing as (N, 3) coordinates"""
    nvox = int(np.prod(shape))
    lin, coords = {}, {}
    for k in class_keys:
        if rng.random() < absent_probability:
            continue
        n = min(int(rng.integers(1, max_n)), nvox)
        v = np.sort(rng.choice(nvox, n, replace=False)).astype(np.uint64)
        lin[k] = v
        coords[k] = np.stack(np.unravel_index(v.astype(np.int64), shape), axis=1)
    return lin, coords


class TestForegroundLocationsStore(unittest.TestCase):
    """Round-trip, edge cases and the interface contract the dataloader relies on."""

    def setUp(self):
        self.rng = np.random.default_rng(1234)
        # int keys, a region key, and an annotated_classes_key style entry
        self.class_keys = [1, 2, 3, (1, 2), (-1, 0, 1, 2, 3)]

    def _build(self, folder, n_cases=37, **writer_kwargs):
        shapes, truth = {}, {}
        w = ForegroundLocationsWriter(folder, self.class_keys, **writer_kwargs)
        for i in range(n_cases):
            cid = f'case_{i}'
            shape = tuple(int(x) for x in self.rng.integers(15, 45, 3))
            lin, coords = _random_runs(self.rng, shape, self.class_keys)
            shapes[cid], truth[cid] = shape, coords
            w.add(cid, shape, lin)
        w.finalize()
        return shapes, truth

    def test_roundtrip_including_tuple_keys(self):
        with TemporaryDirectory() as d:
            shapes, truth = self._build(d)
            store = ForegroundLocations(d)
            # tuple keys must survive the JSON round-trip as tuples, not lists
            self.assertEqual(store.class_keys, self.class_keys)
            for cid, t in truth.items():
                self.assertEqual(set(map(str, store.eligible_classes(cid))), set(map(str, t.keys())))
                self.assertEqual(tuple(store.case_shape(cid)), shapes[cid])
                for k in self.class_keys:
                    expected = t[k] if k in t else np.zeros((0, 3), dtype=np.int64)
                    self.assertEqual(store.count(cid, k), len(expected))
                    np.testing.assert_array_equal(store.all_locations(cid, k), expected)

    def test_sample_returns_a_stored_coordinate(self):
        with TemporaryDirectory() as d:
            _, truth = self._build(d)
            store = ForegroundLocations(d)
            rs = np.random.RandomState(0)
            for cid, t in truth.items():
                for k in t:
                    s = store.sample(cid, k, rng=rs)
                    self.assertEqual(s.shape, (3,))
                    self.assertTrue(np.any(np.all(t[k] == s, axis=1)), f'{cid} {k} -> {s}')

    def test_sampling_is_uniform(self):
        with TemporaryDirectory() as d:
            w = ForegroundLocationsWriter(d, [1])
            shape = (10, 10, 10)
            w.add('c', shape, {1: np.arange(50, dtype=np.uint64)})
            w.finalize()
            store = ForegroundLocations(d)
            rs = np.random.RandomState(0)
            hist = np.zeros(50)
            for _ in range(50 * 400):
                x, y, z = store.sample('c', 1, rng=rs)
                hist[x * 100 + y * 10 + z] += 1
            # all 50 must be hit, and none should be wildly over/under represented
            self.assertTrue((hist > 0).all())
            self.assertLess(np.abs(hist / hist.mean() - 1).max(), 0.35)

    def test_unaligned_tail_and_chunk_boundaries(self):
        # deliberately tiny chunks so that runs straddle many chunk and block boundaries and the
        # final chunk is only partially filled
        with TemporaryDirectory() as d:
            shapes, truth = self._build(d, n_cases=11, chunk_size=64, block_size=32)
            store = ForegroundLocations(d)
            for cid, t in truth.items():
                for k in t:
                    np.testing.assert_array_equal(store.all_locations(cid, k), t[k])

    def test_case_without_any_foreground(self):
        with TemporaryDirectory() as d:
            w = ForegroundLocationsWriter(d, [1, 2])
            w.add('empty', (8, 8, 8), {})
            w.add('nonempty', (8, 8, 8), {1: np.array([5], dtype=np.uint64)})
            w.finalize()
            store = ForegroundLocations(d)
            self.assertEqual(store.eligible_classes('empty'), [])
            self.assertEqual(store.count('empty', 1), 0)
            self.assertEqual(store.eligible_classes('nonempty'), [1])
            with self.assertRaises(KeyError):
                store.sample('empty', 1)

    def test_handles_are_dropped_on_pickling(self):
        with TemporaryDirectory() as d:
            _, truth = self._build(d, n_cases=5)
            store = ForegroundLocations(d)
            cid = next(iter(truth))
            k = next(iter(truth[cid]))
            _ = store.all_locations(cid, k)                      # force handles open
            self.assertIsNotNone(store._handles)
            state = store.__getstate__()
            self.assertIsNone(state['_handles'])                 # not shipped to workers
            revived = pickle.loads(pickle.dumps(store))
            np.testing.assert_array_equal(revived.all_locations(cid, k), store.all_locations(cid, k))

    def test_unknown_case_and_class(self):
        with TemporaryDirectory() as d:
            self._build(d, n_cases=3)
            store = ForegroundLocations(d)
            with self.assertRaises(KeyError):
                store.eligible_classes('does_not_exist')
            self.assertEqual(store.count('case_0', 12345), 0)    # unknown class -> no locations

    def test_factory_falls_back_to_legacy(self):
        with TemporaryDirectory() as d:
            self.assertFalse(has_foreground_locations(d))
            self.assertIsInstance(get_foreground_locations(d, verbose=False), LegacyForegroundLocations)
            self._build(d, n_cases=3)
            self.assertTrue(has_foreground_locations(d))
            self.assertIsInstance(get_foreground_locations(d, verbose=False), ForegroundLocations)

    def test_all_store_files_carry_the_reserved_prefix(self):
        # the rest of nnU-Net identifies cases by scanning this folder, so nothing may look like a case
        with TemporaryDirectory() as d:
            self._build(d, n_cases=3)
            for f in os.listdir(d):
                self.assertTrue(f.startswith(FG_SAMPLING_PREFIX), f)


class TestLegacyAndStoreAgree(unittest.TestCase):
    """The store must be a drop-in replacement for the legacy class_locations dict."""

    def test_same_support_and_interface(self):
        rng = np.random.default_rng(0)
        class_keys = [1, 2, (1, 2)]
        shape = (12, 13, 14)
        with TemporaryDirectory() as d:
            w = ForegroundLocationsWriter(d, class_keys)
            expected = {}
            for i in range(6):
                cid = f'c{i}'
                lin, coords = _random_runs(rng, shape, class_keys, absent_probability=0.4, max_n=300)
                expected[cid] = coords
                w.add(cid, shape, lin)
                # write a legacy pkl with the channel axis prepended, exactly as preprocessing used to
                legacy = {k: (np.concatenate([np.zeros((len(v), 1), np.int64), v], axis=1)
                              if k in coords else []) for k, v in
                          [(kk, coords.get(kk)) for kk in class_keys]}
                with open(os.path.join(d, cid + '.pkl'), 'wb') as f:
                    pickle.dump({'class_locations': legacy}, f)
            w.finalize()

            store, legacy_reader = ForegroundLocations(d), LegacyForegroundLocations(d)
            for cid, coords in expected.items():
                self.assertEqual(set(map(str, store.eligible_classes(cid))),
                                 set(map(str, legacy_reader.eligible_classes(cid))))
                for k in class_keys:
                    self.assertEqual(store.count(cid, k), legacy_reader.count(cid, k))
                    a = store.all_locations(cid, k)
                    b = legacy_reader.all_locations(cid, k)
                    # the store is sorted, the legacy arrays are not -> compare as sets
                    np.testing.assert_array_equal(a, b[np.lexsort((b[:, 2], b[:, 1], b[:, 0]))] if len(b) else b)


class TestSamplingLocationExtraction(unittest.TestCase):
    def test_present_labels_does_not_change_the_result(self):
        rng = np.random.default_rng(3)
        seg = rng.integers(0, 4, (1, 20, 21, 22)).astype(np.int16)
        classes = [1, 2, 3, 5]                      # 5 is not present
        a = DefaultPreprocessor._sample_foreground_locations(seg, classes)
        b = DefaultPreprocessor._sample_foreground_locations(seg, classes, present_labels=[0, 1, 2, 3])
        self.assertEqual(set(map(str, a.keys())), set(map(str, b.keys())))
        for k in a:
            np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))

    def test_ravel_coords_drops_the_channel_axis_and_sorts(self):
        shape = (5, 6, 7)
        coords = np.array([[0, 4, 5, 6], [0, 0, 0, 0], [0, 2, 3, 4]])
        lin = ravel_coords(coords, shape)
        self.assertTrue(np.all(np.diff(lin) > 0))
        np.testing.assert_array_equal(np.stack(np.unravel_index(lin.astype(np.int64), shape), axis=1),
                                      np.array([[0, 0, 0], [2, 3, 4], [4, 5, 6]]))
        # already stripped coordinates are accepted too
        np.testing.assert_array_equal(ravel_coords(coords[:, 1:], shape), lin)

    def test_collect_list_covers_regions_and_ignore_label(self):
        # plain labels
        lm = LabelManager({'background': 0, 'a': 1, 'b': 2}, None)
        self.assertEqual(get_classes_or_regions_to_collect(lm), [1, 2])

        # region based
        lm = LabelManager({'background': 0, 'a': 1, 'ab': (1, 2)}, regions_class_order=(1, 2))
        collected = get_classes_or_regions_to_collect(lm)
        self.assertTrue(lm.has_regions)
        self.assertEqual(collected, list(lm.foreground_regions))

        # ignore label -> the annotated_classes_key must be appended ...
        lm = LabelManager({'background': 0, 'a': 1, 'b': 2, 'ignore': 3}, None)
        def norm(lst):
            return [tuple(int(x) for x in i) if isinstance(i, (tuple, list)) else int(i) for i in lst]
        collected = get_classes_or_regions_to_collect(lm)
        annotated_key = tuple([-1] + [int(i) for i in lm.all_labels])
        self.assertIn(annotated_key, norm(collected))
        # ... without mutating the list the LabelManager owns, so calling it twice is stable
        self.assertEqual(norm(get_classes_or_regions_to_collect(lm)), norm(collected))
        self.assertNotIn(annotated_key, norm(lm.foreground_labels))


class TestDatasetIntegration(unittest.TestCase):
    """The store lives in the same folder as the cases, so it must never look like one."""

    def _make_case(self, folder, name, shape=(1, 8, 9, 10)):
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        rng = np.random.default_rng(0)
        data = rng.random(shape).astype(np.float32)
        seg = rng.integers(0, 3, shape).astype(np.int8)
        nnUNetDatasetBlosc2.save_case(data, seg, {'spacing': [1.0, 1.0, 1.0]}, os.path.join(folder, name),
                                      chunks=(1, 4, 4, 4), blocks=(1, 2, 2, 2),
                                      chunks_seg=(1, 4, 4, 4), blocks_seg=(1, 2, 2, 2))
        return data, seg

    def test_load_case_does_not_return_properties_and_accessors_work(self):
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        with TemporaryDirectory() as d:
            data, seg = self._make_case(d, 'case_a')
            ds = nnUNetDatasetBlosc2(d)
            returned = ds.load_case('case_a')
            self.assertEqual(len(returned), 3)                     # (data, seg, seg_prev)
            d_, s_, prev = returned
            self.assertIsNone(prev)
            np.testing.assert_allclose(d_[:], data, rtol=1e-5)
            self.assertEqual(ds.get_shape('case_a'), tuple(data.shape[1:]))
            self.assertEqual(ds.get_properties('case_a')['spacing'], [1.0, 1.0, 1.0])

    def test_store_files_are_not_mistaken_for_cases(self):
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, infer_dataset_class
        with TemporaryDirectory() as d:
            for name in ('case_a', 'case_b'):
                self._make_case(d, name)
            self.assertEqual(sorted(nnUNetDatasetBlosc2.get_identifiers(d)), ['case_a', 'case_b'])
            self.assertIs(infer_dataset_class(d), nnUNetDatasetBlosc2)

            # now add the store, which contributes .b2nd, .npy and .json files to the same folder
            w = ForegroundLocationsWriter(d, [1, 2])
            w.add('case_a', (8, 9, 10), {1: np.array([3, 17], dtype=np.uint64)})
            w.add('case_b', (8, 9, 10), {2: np.array([5], dtype=np.uint64)})
            w.finalize()

            self.assertEqual(sorted(nnUNetDatasetBlosc2.get_identifiers(d)), ['case_a', 'case_b'])
            # .json would otherwise trip the "exactly one file ending" assertion
            self.assertIs(infer_dataset_class(d), nnUNetDatasetBlosc2)
            ds = nnUNetDatasetBlosc2(d)
            self.assertIsInstance(ds.foreground_locations, ForegroundLocations)
            self.assertEqual(ds.foreground_locations.eligible_classes('case_a'), [1])


if __name__ == '__main__':
    unittest.main()
