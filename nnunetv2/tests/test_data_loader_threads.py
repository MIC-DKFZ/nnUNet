import io
import sys
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
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
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


if __name__ == '__main__':
    unittest.main()
