import io
import unittest
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_module
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader, _writeable
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class TestDataLoaderCropsAreWriteable(unittest.TestCase):
    def test_crops_of_unpacked_data_are_writeable(self):
        # the data augmentation writes into the patches in place. A crop inside an unpacked .npy case is a view of a
        # read-only memory-mapped file, and writing into that is a segmentation fault in the dataloader worker
        with TemporaryDirectory() as folder, TemporaryDirectory() as prev_stage_folder:
            rng = np.random.default_rng(0)
            np.save(join(folder, 'case.npy'), rng.random((2, 20, 20, 20), dtype=np.float32))
            # int16 like the cast below, so that the cast does not copy and the crop stays a view
            np.save(join(folder, 'case_seg.npy'), rng.integers(-1, 3, (1, 20, 20, 20)).astype(np.int16))
            # previous stage segmentations have no channel axis
            np.save(join(prev_stage_folder, 'case.npy'), rng.integers(0, 3, (20, 20, 20)).astype(np.int16))
            with redirect_stdout(io.StringIO()):  # the notice about the missing foreground location store
                ds = nnUNetDatasetNumpy(folder, ['case'], folder_with_segs_from_previous_stage=prev_stage_folder)
            # patch size smaller than the case and no padding: every crop lies inside the case
            dl = nnUNetDataLoader(ds, 2, (8, 8, 8), (8, 8, 8), LabelManager({'background': 0, 'a': 1}, None))

            writeable = []
            from_numpy = torch.from_numpy

            def recording_from_numpy(arr):
                writeable.append(arr.flags.writeable)
                return from_numpy(arr)

            with mock.patch.object(torch, 'from_numpy', recording_from_numpy):
                dl.generate_train_batch()
            self.assertEqual(writeable, [True] * 6)  # 2 samples x (image, segmentation, previous stage)

    def test_only_read_only_arrays_are_copied(self):
        arr = np.arange(6.)
        self.assertIs(_writeable(arr), arr)
        arr.flags.writeable = False
        copied = _writeable(arr)
        self.assertIsNot(copied, arr)
        self.assertTrue(copied.flags.writeable)
        np.testing.assert_array_equal(copied, arr)


class TestOnTrainStartOrder(unittest.TestCase):
    """
    The dataloader workers start reading as soon as get_dataloaders() returns, so the dataloaders must be created
    after rank 0 has unpacked the dataset and, in DDP, after the barrier that makes the other ranks wait for it.
    """

    def _calls(self, global_rank: int, is_ddp: bool):
        calls = []

        def get_dataloaders():
            calls.append('dataloaders')
            return None, None

        with TemporaryDirectory() as preprocessed_folder, TemporaryDirectory() as output_folder:
            save_json({}, join(preprocessed_folder, 'dataset_fingerprint.json'))
            trainer = SimpleNamespace(
                was_initialized=True, output_folder=output_folder, output_folder_base=output_folder,
                preprocessed_dataset_folder=preprocessed_folder, preprocessed_dataset_folder_base=preprocessed_folder,
                enable_deep_supervision=True, device=torch.device('cpu'), global_rank=global_rank, is_ddp=is_ddp,
                plans_manager=SimpleNamespace(plans={}), dataset_json={},
                dataset_class=SimpleNamespace(unpack_dataset=lambda *args, **kwargs: calls.append('unpack')),
                get_dataloaders=get_dataloaders,
                set_deep_supervision_enabled=lambda enabled: None, print_plans=lambda: None,
                plot_network_architecture=lambda: None, _save_debug_information=lambda: None)
            with mock.patch.object(trainer_module, 'dist', SimpleNamespace(barrier=lambda: calls.append('barrier'))):
                nnUNetTrainer.on_train_start(trainer)
        return calls

    def test_single_process(self):
        self.assertEqual(self._calls(global_rank=0, is_ddp=False), ['unpack', 'dataloaders'])

    def test_ddp_rank_0(self):
        self.assertEqual(self._calls(global_rank=0, is_ddp=True), ['unpack', 'barrier', 'dataloaders'])

    def test_ddp_other_ranks(self):
        self.assertEqual(self._calls(global_rank=1, is_ddp=True), ['barrier', 'dataloaders'])


if __name__ == '__main__':
    unittest.main()
