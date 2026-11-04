import multiprocessing
import os
import types
import unittest
import unittest.mock
from tempfile import TemporaryDirectory

import blosc2
import numpy as np

from nnunetv2.training.dataloading import nnunet_dataset
from nnunetv2.training.dataloading.nnunet_dataset import (
    BLOSC2_READ_AHEAD_THREADS, enable_blosc2_read_ahead, nnUNetDatasetBlosc2)


def _env_without_blosc_prefetch():
    return {k: v for k, v in os.environ.items() if k != 'BLOSC_PREFETCH'}


def _read_ahead_threads_after_load_case(folder):
    # runs in a fresh process: load a case, then read back the setting (set_prefetch returns the previous value)
    nnUNetDatasetBlosc2(folder).load_case('c')
    threads = blosc2.set_prefetch(0)
    return threads


class TestBlosc2ReadAhead(unittest.TestCase):
    def setUp(self):
        nnunet_dataset._blosc2_read_ahead_pid = None

    def tearDown(self):
        nnunet_dataset._blosc2_read_ahead_pid = None

    def test_does_nothing_without_set_prefetch(self):
        # blosc2 versions before read-ahead have no set_prefetch
        old_blosc2 = types.SimpleNamespace()
        with unittest.mock.patch.object(nnunet_dataset, 'blosc2', old_blosc2), \
                unittest.mock.patch.dict(os.environ, _env_without_blosc_prefetch(), clear=True):
            enable_blosc2_read_ahead()

    def test_turns_it_on_once_per_process(self):
        set_prefetch = unittest.mock.Mock(return_value=0)
        with unittest.mock.patch.object(blosc2, 'set_prefetch', set_prefetch, create=True), \
                unittest.mock.patch.dict(os.environ, _env_without_blosc_prefetch(), clear=True):
            enable_blosc2_read_ahead()
            enable_blosc2_read_ahead()
            set_prefetch.assert_called_once_with(BLOSC2_READ_AHEAD_THREADS)
            # a forked worker has another pid and must set it again
            with unittest.mock.patch.object(nnunet_dataset.os, 'getpid', return_value=os.getpid() + 1):
                enable_blosc2_read_ahead()
            self.assertEqual(set_prefetch.call_count, 2)

    def test_blosc_prefetch_takes_precedence(self):
        set_prefetch = unittest.mock.Mock(return_value=0)
        for value in ('0', '1', '8'):
            nnunet_dataset._blosc2_read_ahead_pid = None
            with unittest.mock.patch.object(blosc2, 'set_prefetch', set_prefetch, create=True), \
                    unittest.mock.patch.dict(os.environ, {'BLOSC_PREFETCH': value}):
                enable_blosc2_read_ahead()
        set_prefetch.assert_not_called()

    def test_failing_set_prefetch_only_warns(self):
        # an unexpected set_prefetch (e.g. a changed signature) must not stop data loading
        set_prefetch = unittest.mock.Mock(side_effect=TypeError('unexpected signature'))
        with unittest.mock.patch.object(blosc2, 'set_prefetch', set_prefetch, create=True), \
                unittest.mock.patch.dict(os.environ, _env_without_blosc_prefetch(), clear=True):
            with self.assertWarns(UserWarning):
                enable_blosc2_read_ahead()
            enable_blosc2_read_ahead()  # once per process: no second attempt, no second warning
        set_prefetch.assert_called_once_with(BLOSC2_READ_AHEAD_THREADS)

    def test_load_case_turns_it_on(self):
        set_prefetch = unittest.mock.Mock(return_value=0)
        with TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            data = rng.random((1, 8, 9, 10)).astype(np.float32)
            seg = rng.integers(0, 3, (1, 8, 9, 10)).astype(np.int8)
            nnUNetDatasetBlosc2.save_case(data, seg, {'spacing': [1.0, 1.0, 1.0]}, os.path.join(d, 'c'),
                                          chunks=(1, 4, 4, 4), blocks=(1, 2, 2, 2),
                                          chunks_seg=(1, 4, 4, 4), blocks_seg=(1, 2, 2, 2))
            with unittest.mock.patch.object(blosc2, 'set_prefetch', set_prefetch, create=True), \
                    unittest.mock.patch.dict(os.environ, _env_without_blosc_prefetch(), clear=True):
                loaded_data, loaded_seg, seg_prev = nnUNetDatasetBlosc2(d).load_case('c')
                np.testing.assert_array_equal(loaded_data[:], data)
                np.testing.assert_array_equal(loaded_seg[:], seg)
                self.assertIsNone(seg_prev)
        set_prefetch.assert_called_once_with(BLOSC2_READ_AHEAD_THREADS)

    @unittest.skipUnless(hasattr(blosc2, 'set_prefetch'), 'this blosc2 has no read-ahead')
    def test_real_blosc2_in_a_new_process(self):
        # with a blosc2 that has read-ahead, a spawned worker that loads a case reads ahead with 4 threads,
        # unless BLOSC_PREFETCH says otherwise
        with TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            nnUNetDatasetBlosc2.save_case(rng.random((1, 8, 9, 10)).astype(np.float32),
                                          rng.integers(0, 3, (1, 8, 9, 10)).astype(np.int8),
                                          {'spacing': [1.0, 1.0, 1.0]}, os.path.join(d, 'c'),
                                          chunks=(1, 4, 4, 4), blocks=(1, 2, 2, 2),
                                          chunks_seg=(1, 4, 4, 4), blocks_seg=(1, 2, 2, 2))
            ctx = multiprocessing.get_context('spawn')
            for env, expected in ((None, BLOSC2_READ_AHEAD_THREADS), ('0', 0), ('2', 2)):
                patched = _env_without_blosc_prefetch()
                if env is not None:
                    patched['BLOSC_PREFETCH'] = env
                with unittest.mock.patch.dict(os.environ, patched, clear=True):
                    with ctx.Pool(1) as pool:
                        self.assertEqual(pool.apply(_read_ahead_threads_after_load_case, (d,)), expected)


if __name__ == '__main__':
    unittest.main()
