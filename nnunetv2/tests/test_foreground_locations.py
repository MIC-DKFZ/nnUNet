import io
import multiprocessing
import os
import pickle
import unittest
import unittest.mock
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory

import numpy as np

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.sampling_locations.extract_sampling_locations import (
    present_labels_from_class_locations)
from nnunetv2.training.dataloading.foreground_locations import (
    FG_SAMPLING_DIRNAME, ForegroundLocations, ForegroundLocationsWriter, LegacyForegroundLocations,
    ForegroundLocationsWriter, announce_missing_store, get_foreground_locations,
    has_foreground_locations, ravel_coords)
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


def _announce_in_child(folder):
    """Module level so that it survives pickling to a 'spawn' worker."""
    from nnunetv2.training.dataloading.foreground_locations import announce_missing_store
    return announce_missing_store(folder)


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
            self.assertIsInstance(get_foreground_locations(d), LegacyForegroundLocations)
            self._build(d, n_cases=3)
            self.assertTrue(has_foreground_locations(d))
            self.assertIsInstance(get_foreground_locations(d), ForegroundLocations)

    def test_store_is_confined_to_its_own_subfolder(self):
        # the rest of nnU-Net identifies cases by scanning the configuration folder, so the store must not
        # drop any file into it
        with TemporaryDirectory() as d:
            self._build(d, n_cases=3)
            self.assertEqual(os.listdir(d), [FG_SAMPLING_DIRNAME])
            self.assertTrue(os.path.isdir(os.path.join(d, FG_SAMPLING_DIRNAME)))

    def test_rebuilding_clears_the_previous_store(self):
        with TemporaryDirectory() as d:
            self._build(d, n_cases=3)
            stale = os.path.join(d, FG_SAMPLING_DIRNAME, 'stale_from_an_older_version.npy')
            np.save(stale, np.zeros(3))
            self._build(d, n_cases=2)
            self.assertFalse(os.path.isfile(stale))
            self.assertEqual(len(ForegroundLocations(d).identifiers), 2)


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

    def test_present_labels_from_legacy_class_locations(self):
        # an entry is empty exactly when none of its labels are present, so only labels that appear in an
        # empty entry and in no non-empty one may be ruled out
        nonempty = np.zeros((3, 4), dtype=np.int64)
        cl = {1: nonempty, 2: [], 3: nonempty}
        self.assertEqual(present_labels_from_class_locations(cl, [1, 2, 3]), [1, 3])

        # a label shared between an empty and a non-empty region must survive
        cl = {(1, 2): [], (2, 3): nonempty}
        self.assertEqual(present_labels_from_class_locations(cl, [(1, 2), (2, 3)]), [2, 3])

        # a requested label the legacy dict says nothing about (dataset.json changed) is kept
        cl = {1: []}
        self.assertEqual(present_labels_from_class_locations(cl, [1, 7]), [7])

        # the ignore-label pseudo class carries -1, which is harmless and must not break anything
        cl = {(-1, 0, 1): nonempty, 2: []}
        self.assertEqual(present_labels_from_class_locations(cl, [(-1, 0, 1), 2]), [-1, 0, 1])

    def test_legacy_hint_does_not_change_the_sampling_result(self):
        # migrating a legacy dataset must produce exactly what a hint-free run produces
        rng = np.random.default_rng(7)
        seg = rng.integers(0, 4, (1, 18, 19, 20)).astype(np.int16)
        seg[seg == 3] = 0                                   # label 3 absent, 5 never existed
        classes = [1, 2, 3, 5]
        legacy = DefaultPreprocessor._sample_foreground_locations(seg, classes)
        hint = present_labels_from_class_locations(legacy, classes)
        self.assertEqual(hint, [1, 2])                      # 3 and 5 ruled out from the legacy result alone
        with_hint = DefaultPreprocessor._sample_foreground_locations(seg, classes, present_labels=hint)
        self.assertEqual(set(map(str, legacy.keys())), set(map(str, with_hint.keys())))
        for k in legacy:
            np.testing.assert_array_equal(np.asarray(legacy[k]), np.asarray(with_hint[k]))

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
        self.assertEqual(lm.classes_or_regions_for_sampling, [1, 2])

        # region based
        lm = LabelManager({'background': 0, 'a': 1, 'ab': (1, 2)}, regions_class_order=(1, 2))
        collected = lm.classes_or_regions_for_sampling
        self.assertTrue(lm.has_regions)
        self.assertEqual(collected, list(lm.foreground_regions))

        # ignore label -> the annotated_classes_key must be appended ...
        lm = LabelManager({'background': 0, 'a': 1, 'b': 2, 'ignore': 3}, None)
        def norm(lst):
            return [tuple(int(x) for x in i) if isinstance(i, (tuple, list)) else int(i) for i in lst]
        collected = lm.classes_or_regions_for_sampling
        self.assertIn(lm.annotated_classes_key, norm(collected))
        # ... without mutating the list the LabelManager owns, so calling it twice is stable
        self.assertEqual(norm(lm.classes_or_regions_for_sampling), norm(collected))
        self.assertNotIn(lm.annotated_classes_key, norm(lm.foreground_labels))

    def test_dataloader_and_extraction_agree_on_the_annotated_classes_key(self):
        # the producer writes this key into the store, the dataloader looks it up by value
        lm = LabelManager({'background': 0, 'a': 1, 'b': 2, 'ignore': 3}, None)
        self.assertEqual(lm.classes_or_regions_for_sampling[-1], lm.annotated_classes_key)
        self.assertEqual(lm.annotated_classes_key, tuple([-1] + [int(i) for i in lm.all_labels]))


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

            # now add the store, which contributes .b2nd, .npy and .json files of its own
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


class TestInMemoryAndSpilledStoresAgree(unittest.TestCase):
    """
    The writer builds the array in RAM and serialises it once; oversized stores spill to the
    incremental on-disk path. Both must produce exactly the same store.
    """

    def _build(self, folder, max_in_memory_bytes):
        rng = np.random.default_rng(7)
        class_keys = [1, 2, (1, 2)]
        shape = (40, 41, 42)
        w = ForegroundLocationsWriter(folder, class_keys, max_in_memory_bytes=max_in_memory_bytes)
        truth = {}
        for i in range(25):
            cid = f'c{i}'
            lin, coords = _random_runs(rng, shape, class_keys, absent_probability=0.2, max_n=4000)
            truth[cid] = coords
            w.add(cid, shape, lin)
        w.finalize()
        return truth, w._on_disk

    def test_spilled_store_matches_in_memory_store(self):
        with TemporaryDirectory() as d1, TemporaryDirectory() as d2:
            truth_mem, spilled_mem = self._build(d1, max_in_memory_bytes=10 ** 12)   # never spills
            truth_disk, spilled_disk = self._build(d2, max_in_memory_bytes=1)        # spills immediately
            self.assertFalse(spilled_mem)
            self.assertTrue(spilled_disk, 'the spill path was never taken, so it is untested')

            a, b = ForegroundLocations(d1), ForegroundLocations(d2)
            self.assertEqual(a.identifiers, b.identifiers)
            self.assertEqual(a.class_keys, b.class_keys)
            for cid in a.identifiers:
                self.assertEqual(a.eligible_classes(cid), b.eligible_classes(cid))
                for k in a.class_keys:
                    np.testing.assert_array_equal(a.all_locations(cid, k), b.all_locations(cid, k))
                    np.testing.assert_array_equal(a.all_locations(cid, k),
                                                  truth_mem[cid].get(k, np.zeros((0, 3), np.int64)))

    def test_in_memory_store_is_a_valid_blosc2_frame(self):
        # finalize() writes the frame with to_cframe() rather than through blosc2's own file
        # handling, so check it reopens and that the fast read path works on it
        with TemporaryDirectory() as d:
            self._build(d, max_in_memory_bytes=10 ** 12)
            store = ForegroundLocations(d)
            cid = store.identifiers[0]
            k = store.eligible_classes(cid)[0]
            self.assertEqual(store.sample(cid, k).shape, (3,))
            self.assertGreater(store.count(cid, k), 0)


class TestAllBlosc2ReadsAreMemoryMapped(unittest.TestCase):
    """
    Reads must be memory mapped so that dataloader workers share physical pages instead of each
    holding a copy, and so blosc2's chunk offset table is paged in lazily. This is easy to drop by
    accident when adding a new read, hence the source level check.
    """

    MODULES = ['nnunetv2/training/dataloading/foreground_locations.py',
               'nnunetv2/training/dataloading/nnunet_dataset.py',
               'nnunetv2/preprocessing/sampling_locations/extract_sampling_locations.py']

    @staticmethod
    def _call_texts(src, needle='blosc2.open('):
        """Yield the full text of each blosc2.open(...) call, following it across line breaks."""
        out, i = [], src.find(needle)
        while i != -1:
            j, depth = i + len(needle) - 1, 0
            while j < len(src):
                if src[j] == '(':
                    depth += 1
                elif src[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append(src[i:j + 1])
            i = src.find(needle, j)
        return out

    def test_every_blosc2_open_passes_mmap_kwargs(self):
        import nnunetv2
        root = os.path.dirname(os.path.dirname(os.path.abspath(nnunetv2.__file__)))
        checked = 0
        for rel in self.MODULES:
            path = os.path.join(root, rel)
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                src = f.read()
            for call in self._call_texts(src):
                checked += 1
                self.assertTrue('MMAP_KWARGS' in call or 'mmap_kwargs' in call,
                                f'blosc2.open without mmap kwargs in {rel}:\n{call}')
        self.assertGreater(checked, 0, 'found no blosc2.open calls to check - did the modules move?')

    def test_mmap_kwargs_is_enabled_off_windows(self):
        from nnunetv2.training.dataloading.foreground_locations import MMAP_KWARGS
        if os.name == 'nt':
            self.assertEqual(MMAP_KWARGS, {})
        else:
            self.assertEqual(MMAP_KWARGS, {'mmap_mode': 'r'})

    def test_store_and_index_open_without_error_and_read_back(self):
        # the functional counterpart: whatever the kwargs are, reads must still work
        with TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            w = ForegroundLocationsWriter(d, [1, 2])
            lin = np.sort(rng.choice(9999, 500, replace=False)).astype(np.uint64)
            w.add('c', (20, 20, 25), {1: lin})
            w.finalize()
            store = ForegroundLocations(d)
            np.testing.assert_array_equal(
                store.all_locations('c', 1),
                np.stack(np.unravel_index(lin.astype(np.int64), (20, 20, 25)), axis=1))


class TestMissingStoreIsAnnouncedExactlyOnce(unittest.TestCase):
    """
    The notice must appear once per training run, not once per dataloader worker. Workers inherit a
    copy of this module's globals, so a module-level "already warned" set is not enough on its own.
    """

    def setUp(self):
        from nnunetv2.training.dataloading import foreground_locations as fl
        fl._ANNOUNCED_FOLDERS.clear()

    def test_announced_once_per_folder(self):
        with TemporaryDirectory() as d:
            self.assertTrue(announce_missing_store(d))      # first time: printed
            self.assertFalse(announce_missing_store(d))     # same folder again: quiet
            self.assertFalse(announce_missing_store(d))

    def test_silent_when_a_store_exists(self):
        with TemporaryDirectory() as d:
            w = ForegroundLocationsWriter(d, [1])
            w.add('c', (4, 4, 4), {1: np.array([0], dtype=np.uint64)})
            w.finalize()
            self.assertFalse(announce_missing_store(d))

    def test_silent_in_worker_processes(self):
        # the actual regression: every dataloader worker used to print the notice again
        for method in ('fork', 'spawn'):
            if method not in multiprocessing.get_all_start_methods():
                continue
            with self.subTest(start_method=method), TemporaryDirectory() as d:
                ctx = multiprocessing.get_context(method)
                with ctx.Pool(2) as p:
                    printed = p.map(_announce_in_child, [d, d, d, d])
                self.assertEqual(printed, [False] * 4,
                                 f'a {method} worker announced the missing store')
            # and the main process still announces it for that folder
            with TemporaryDirectory() as d2:
                self.assertTrue(announce_missing_store(d2))

    def test_silent_on_secondary_ddp_ranks(self):
        with TemporaryDirectory() as d:
            with unittest.mock.patch.dict(os.environ, {'LOCAL_RANK': '1'}):
                self.assertFalse(announce_missing_store(d))
            self.assertTrue(announce_missing_store(d))      # rank 0 still announces

    def test_dataset_announces_at_construction_not_on_first_use(self):
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        with TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            nnUNetDatasetBlosc2.save_case(rng.random((1, 4, 4, 4)).astype(np.float32),
                                          rng.integers(0, 2, (1, 4, 4, 4)).astype(np.int8),
                                          {'spacing': [1.0, 1.0, 1.0]}, os.path.join(d, 'c'),
                                          chunks=(1, 2, 2, 2), blocks=(1, 2, 2, 2),
                                          chunks_seg=(1, 2, 2, 2), blocks_seg=(1, 2, 2, 2))
            buf = io.StringIO()
            with redirect_stdout(buf):
                ds = nnUNetDatasetBlosc2(d)
            self.assertIn('no foreground sampling location store', buf.getvalue())
            # constructing more datasets on the same folder (the validation loop does this per case)
            # and touching the property must stay quiet
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                _ = ds.foreground_locations
                for _ in range(3):
                    _ = nnUNetDatasetBlosc2(d).foreground_locations
            self.assertEqual(buf2.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
