import io
import os
import sys
import threading
import types
import unittest
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join
from threadpoolctl import threadpool_info

import nnunetv2.training.dataloading.data_loader as data_loader_module
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, nnUNetDatasetNumpy
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def _make_loader(folder, transforms=None):
    rng = np.random.default_rng(0)
    for c in ('a', 'b'):
        np.save(join(folder, f'{c}.npy'), rng.random((1, 12, 12, 12), dtype=np.float32))
        np.save(join(folder, f'{c}_seg.npy'), rng.integers(0, 2, (1, 12, 12, 12)).astype(np.int16))
    with redirect_stdout(io.StringIO()):  # the notice about the missing foreground location store
        ds = nnUNetDatasetNumpy(folder, ['a', 'b'])
    return nnUNetDataLoader(ds, 2, (8, 8, 8), (8, 8, 8), LabelManager({'background': 0, 'a': 1}, None),
                            transforms=transforms)


class TestDataLoaderThreadLimits(unittest.TestCase):
    def setUp(self):
        data_loader_module._threadpool_controller = None

    def test_math_libraries_are_single_threaded_during_a_batch(self):
        seen = []

        def transforms(image, segmentation):
            seen.append([p['num_threads'] for p in threadpool_info()])
            return {'image': image, 'segmentation': segmentation}

        before = [p['num_threads'] for p in threadpool_info()]
        with TemporaryDirectory() as folder:
            dl = _make_loader(folder, transforms)
            for _ in range(3):
                dl.generate_train_batch()
        self.assertEqual(len(seen), 6)
        for limits in seen:
            self.assertTrue(limits)
            self.assertTrue(all(n == 1 for n in limits), limits)
        # the limits are lifted again after every batch
        self.assertEqual([p['num_threads'] for p in threadpool_info()], before)

    @unittest.skipIf(data_loader_module.ThreadpoolController is None, 'threadpoolctl < 3.0')
    def test_one_controller_per_process_until_new_modules_are_imported(self):
        real = data_loader_module.ThreadpoolController
        with TemporaryDirectory() as folder, \
                mock.patch.object(data_loader_module, 'ThreadpoolController', side_effect=real) as controller:
            dl = _make_loader(folder)
            for _ in range(3):
                dl.generate_train_batch()
            self.assertEqual(controller.call_count, 1)
            # a newly imported module may have brought its own thread pool: look at the libraries again
            sys.modules['_nnunet_test_dummy_module'] = types.ModuleType('_nnunet_test_dummy_module')
            try:
                dl.generate_train_batch()
                dl.generate_train_batch()
            finally:
                del sys.modules['_nnunet_test_dummy_module']
            self.assertEqual(controller.call_count, 2)



def _make_cascade_loaders(folder, prev_folder):
    """numpy and blosc2 datasets of the same cases, with segmentations from a previous stage; the patch is larger
    than the cases along the first axis, so every crop is padded there"""
    rng = np.random.default_rng(1)
    ids = ['a', 'b', 'c']
    np_folder, b2_folder = join(folder, 'np'), join(folder, 'b2')
    for f in (np_folder, b2_folder):
        os.makedirs(f)
    for c in ids:
        data = rng.random((2, 6, 14, 13), dtype=np.float32)
        seg = rng.integers(-1, 3, (1, 6, 14, 13)).astype(np.int16)
        np.save(join(np_folder, f'{c}.npy'), data)
        np.save(join(np_folder, f'{c}_seg.npy'), seg)
        nnUNetDatasetBlosc2.save_case(data, seg, {}, join(b2_folder, c), chunks=(1, 4, 8, 8), blocks=(1, 2, 4, 4),
                                      chunks_seg=(1, 4, 8, 8), blocks_seg=(1, 2, 4, 4))
        prev = rng.integers(0, 3, (6, 14, 13)).astype(np.int16)
        np.save(join(prev_folder, f'{c}.npy'), prev)
        nnUNetDatasetBlosc2.save_seg(prev, join(prev_folder, c), chunks_seg=(4, 8, 8), blocks_seg=(2, 4, 4))
    lm = LabelManager({'background': 0, 'a': 1, 'b': 2}, None)
    loaders = []
    for cls, f in ((nnUNetDatasetNumpy, np_folder), (nnUNetDatasetBlosc2, b2_folder)):
        with redirect_stdout(io.StringIO()):
            ds = cls(f, ids, folder_with_segs_from_previous_stage=prev_folder)
        loaders.append(nnUNetDataLoader(ds, 3, (8, 10, 10), (8, 10, 10), lm))
    return loaders


def _crop_segs_inline(seg, seg_prev, bbox):
    done = data_loader_module.Future()
    done.set_result(data_loader_module._crop_segs(seg, seg_prev, bbox))
    return done


class TestSegCropThread(unittest.TestCase):
    def test_batches_equal_cropping_in_one_thread(self):
        with TemporaryDirectory() as folder, TemporaryDirectory() as prev_folder:
            for dl in _make_cascade_loaders(folder, prev_folder):
                for seed in range(4):
                    np.random.seed(seed)
                    threaded = dl.generate_train_batch()
                    np.random.seed(seed)
                    with mock.patch.object(data_loader_module, '_crop_segs_async', _crop_segs_inline):
                        inline = dl.generate_train_batch()
                    self.assertEqual(list(threaded['keys']), list(inline['keys']))
                    self.assertEqual(threaded['target'].dtype, inline['target'].dtype)
                    self.assertEqual(tuple(threaded['target'].shape), (3, 2, 8, 10, 10))
                    self.assertTrue(np.array_equal(threaded['data'].numpy(), inline['data'].numpy()))
                    self.assertTrue(np.array_equal(threaded['target'].numpy(), inline['target'].numpy()))
                    # padding of the segmentations is -1
                    self.assertTrue((threaded['target'].numpy() == -1).any())

    def test_segmentations_are_cropped_in_the_seg_crop_thread(self):
        threads = []
        crop = data_loader_module.crop_and_pad_nd

        def recording_crop(image, bbox, pad_value=0, *args, **kwargs):
            threads.append((pad_value, threading.current_thread().name))
            return crop(image, bbox, pad_value, *args, **kwargs)

        with TemporaryDirectory() as folder, TemporaryDirectory() as prev_folder, \
                mock.patch.object(data_loader_module, 'crop_and_pad_nd', recording_crop):
            for dl in _make_cascade_loaders(folder, prev_folder):
                dl.generate_train_batch()
        data_threads = {t for v, t in threads if v == 0}
        seg_threads = {t for v, t in threads if v == -1}
        self.assertEqual(len(threads), 2 * 3 * 3)  # 2 loaders x 3 samples x (data, seg, previous stage seg)
        self.assertEqual(data_threads, {threading.current_thread().name})
        self.assertEqual(len(seg_threads), 1)
        self.assertTrue(next(iter(seg_threads)).startswith('nnUNet_seg_crop'))

    def test_errors_of_the_seg_crop_reach_the_caller(self):
        def failing(*args, **kwargs):
            raise ValueError('seg crop failed')

        with TemporaryDirectory() as folder:
            dl = _make_loader(folder)
            with mock.patch.object(data_loader_module, '_crop_segs', failing):
                with self.assertRaisesRegex(ValueError, 'seg crop failed'):
                    dl.generate_train_batch()


if __name__ == '__main__':
    unittest.main()
