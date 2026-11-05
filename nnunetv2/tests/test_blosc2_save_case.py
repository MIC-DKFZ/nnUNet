import errno
import os
import subprocess
import sys
import unittest
import unittest.mock
from tempfile import TemporaryDirectory

import blosc2
import numpy as np

import nnunetv2
from nnunetv2.training.dataloading import nnunet_dataset
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

CHUNKS, BLOCKS = (1, 8, 16, 16), (1, 4, 8, 8)


def _case(seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((2, 20, 24, 28)).astype(np.float32)
    seg = rng.integers(-1, 4, (1, 20, 24, 28)).astype(np.int16)
    return data, seg


def _save_case(data, seg, folder, name='c'):
    nnUNetDatasetBlosc2.save_case(data, seg, {'spacing': [1.0, 1.0, 1.0]}, os.path.join(folder, name),
                                  chunks=CHUNKS, blocks=BLOCKS)


def _quota_error():
    return OSError(errno.EDQUOT, os.strerror(errno.EDQUOT))


class _FailingFile:
    """A real file whose write or close fails the way a full quota on a network filesystem does."""

    def __init__(self, f, fail_on):
        self._f, self._fail_on = f, fail_on

    def write(self, b):
        if self._fail_on == 'write':
            raise _quota_error()
        return self._f.write(b)

    def flush(self):
        self._f.flush()

    def fileno(self):
        return self._f.fileno()

    def close(self):
        self._f.close()
        if self._fail_on == 'close':
            raise _quota_error()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _failing_open(fail_on, path_suffix):
    def fake_open(path, *args, **kwargs):
        f = open(path, *args, **kwargs)
        return _FailingFile(f, fail_on) if str(path).endswith(path_suffix) else f
    return unittest.mock.patch.object(nnunet_dataset, 'open', fake_open, create=True)


class TestBlosc2SaveCase(unittest.TestCase):
    def test_round_trip(self):
        data, seg = _case()
        with TemporaryDirectory() as d:
            _save_case(data, seg, d)
            self.assertEqual(sorted(os.listdir(d)), ['c.b2nd', 'c.pkl', 'c_seg.b2nd'])
            loaded_data, loaded_seg, _ = nnUNetDatasetBlosc2(d).load_case('c')
            np.testing.assert_array_equal(loaded_data[:], data)
            np.testing.assert_array_equal(loaded_seg[:], seg)
            self.assertEqual(loaded_data.dtype, data.dtype)
            self.assertEqual(loaded_seg.dtype, seg.dtype)
            for arr in (loaded_data, loaded_seg):
                self.assertEqual(arr.chunks, CHUNKS)
                self.assertEqual(arr.blocks, BLOCKS)
                self.assertEqual(arr.schunk.cparams.codec, blosc2.Codec.LZ4HC)
                self.assertEqual(arr.schunk.cparams.clevel, 5)
            self.assertEqual(nnUNetDatasetBlosc2(d).get_properties('c'), {'spacing': [1.0, 1.0, 1.0]})

    def test_same_array_as_blosc2_writes_it(self):
        # what the loader sees is the same as with blosc2.asarray(urlpath=...), which save_case used before
        data, _ = _case()
        cparams = {'codec': blosc2.Codec.LZ4HC, 'clevel': 5, 'filters': [blosc2.Filter.SHUFFLE], 'nthreads': 4}
        with TemporaryDirectory() as d:
            nnunet_dataset.write_b2nd(data, os.path.join(d, 'new.b2nd'), CHUNKS, BLOCKS, cparams)
            blosc2.asarray(data, urlpath=os.path.join(d, 'ref.b2nd'), chunks=CHUNKS, blocks=BLOCKS, cparams=cparams)
            new, ref = (blosc2.open(os.path.join(d, f), mode='r', dparams={'nthreads': 1})
                        for f in ('new.b2nd', 'ref.b2nd'))
            for attr in ('shape', 'dtype', 'chunks', 'blocks'):
                self.assertEqual(getattr(new, attr), getattr(ref, attr))
            for attr in ('codec', 'clevel', 'filters', 'filters_meta', 'typesize', 'splitmode'):
                self.assertEqual(getattr(new.schunk.cparams, attr), getattr(ref.schunk.cparams, attr))
            self.assertEqual(new.schunk.meta['b2nd'], ref.schunk.meta['b2nd'])
            np.testing.assert_array_equal(new[:], ref[:])

    def test_failed_write_raises_and_leaves_no_file(self):
        data, seg = _case()
        for fail_on in ('write', 'close'):
            for failing_file in ('c.b2nd', 'c_seg.b2nd'):
                with self.subTest(fail_on=fail_on, failing_file=failing_file), TemporaryDirectory() as d:
                    with _failing_open(fail_on, os.sep + failing_file + '.tmp'):
                        with self.assertRaises(OSError) as cm:
                            _save_case(data, seg, d)
                    self.assertEqual(cm.exception.errno, errno.EDQUOT)
                    files = os.listdir(d)
                    self.assertNotIn(failing_file, files)
                    self.assertNotIn('c.pkl', files)
                    self.assertEqual([f for f in files if f.endswith('.tmp')], [])

    def test_failed_fsync_raises_and_leaves_no_file(self):
        data, seg = _case()
        with TemporaryDirectory() as d:
            with unittest.mock.patch.object(nnunet_dataset.os, 'fsync', side_effect=_quota_error()):
                with self.assertRaises(OSError):
                    _save_case(data, seg, d)
            self.assertEqual(os.listdir(d), [])

    def test_save_case_replaces_existing_files(self):
        with TemporaryDirectory() as d:
            _save_case(*_case(0), d)
            data, seg = _case(1)
            _save_case(data, seg, d)
            loaded_data, loaded_seg, _ = nnUNetDatasetBlosc2(d).load_case('c')
            np.testing.assert_array_equal(loaded_data[:], data)
            np.testing.assert_array_equal(loaded_seg[:], seg)

    def test_save_seg(self):
        _, seg = _case()
        seg = seg[0].astype(np.uint8)
        with TemporaryDirectory() as d:
            out = os.path.join(d, 'pred')
            nnUNetDatasetBlosc2.save_seg(np.zeros_like(seg), out, chunks_seg=CHUNKS[1:], blocks_seg=BLOCKS[1:])
            nnUNetDatasetBlosc2.save_seg(seg, out, chunks_seg=CHUNKS[1:], blocks_seg=BLOCKS[1:])
            self.assertEqual(os.listdir(d), ['pred.b2nd'])
            loaded = blosc2.open(out + '.b2nd', mode='r')
            np.testing.assert_array_equal(loaded[:], seg)
            self.assertEqual(loaded.chunks, CHUNKS[1:])
            self.assertEqual(loaded.blocks, BLOCKS[1:])

    def test_failed_save_seg_leaves_no_file(self):
        # not even the segmentation an earlier run wrote to the same name
        _, seg = _case()
        seg = seg[0].astype(np.uint8)
        with TemporaryDirectory() as d:
            out = os.path.join(d, 'pred')
            nnUNetDatasetBlosc2.save_seg(seg, out, chunks_seg=CHUNKS[1:], blocks_seg=BLOCKS[1:])
            with _failing_open('close', '.tmp'), self.assertRaises(OSError):
                nnUNetDatasetBlosc2.save_seg(seg, out, chunks_seg=CHUNKS[1:], blocks_seg=BLOCKS[1:])
            self.assertEqual(os.listdir(d), [])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'uses RLIMIT_FSIZE and SIGXFSZ')
    def test_real_write_error(self):
        # A file size limit makes the kernel fail the write with EFBIG, as a full disk or quota fails it with
        # ENOSPC / EDQUOT. Runs in a child process so that the limit cannot affect anything else.
        child = (
            'import os, resource, signal, sys\n'
            'import numpy as np\n'
            'from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2\n'
            'rng = np.random.default_rng(0)\n'
            'data = rng.random((1, 32, 32, 32)).astype(np.float32)\n'
            'seg = rng.integers(0, 3, (1, 32, 32, 32)).astype(np.int16)\n'
            'signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n'
            'resource.setrlimit(resource.RLIMIT_FSIZE, (4096, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))\n'
            'try:\n'
            '    nnUNetDatasetBlosc2.save_case(data, seg, {}, os.path.join(sys.argv[1], "c"),\n'
            '                                  chunks=(1, 16, 16, 16), blocks=(1, 8, 8, 8))\n'
            'except OSError as e:\n'
            '    print("errno", e.errno)\n'
            'else:\n'
            '    print("no error")\n'
        )
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(
            [os.path.dirname(os.path.dirname(nnunetv2.__file__))] + env.get('PYTHONPATH', '').split(os.pathsep))
        with TemporaryDirectory() as d:
            out = subprocess.run([sys.executable, '-c', child, d], capture_output=True, text=True, env=env)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.split(), ['errno', str(errno.EFBIG)], out.stderr)
            self.assertEqual(os.listdir(d), [])


if __name__ == '__main__':
    unittest.main()
