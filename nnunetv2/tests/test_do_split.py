import os
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_module
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

IDENTIFIERS = [f'case_{i:03d}' for i in range(20)]


class _FakeDataset:
    def __init__(self, folder, identifiers=None, folder_with_segs_from_previous_stage=None):
        self.identifiers = IDENTIFIERS if identifiers is None else identifiers

    @staticmethod
    def get_identifiers(folder):
        return IDENTIFIERS


def _fake_trainer(folder: str, fold: int, global_rank: int = 0, is_ddp: bool = False):
    # do_split only needs these attributes, so we don't have to set up plans and preprocessed data
    return SimpleNamespace(dataset_class=_FakeDataset, preprocessed_dataset_folder=folder,
                           preprocessed_dataset_folder_base=folder, folder_with_segs_from_previous_stage=None,
                           fold=fold, global_rank=global_rank, is_ddp=is_ddp, print_to_log_file=lambda *a, **k: None)


def _ddp_worker(rank: int, world_size: int, folder: str):
    dist.init_process_group('gloo', init_method=f"file://{join(folder, 'pg_init')}", rank=rank, world_size=world_size)
    try:
        with mock.patch.object(trainer_module, 'save_json', wraps=trainer_module.save_json) as save_spy, \
                mock.patch.object(trainer_module, 'load_json', wraps=trainer_module.load_json) as load_spy:
            tr_keys, val_keys = nnUNetTrainer.do_split(_fake_trainer(folder, 2, rank, True))
        save_json({'tr': list(tr_keys), 'val': list(val_keys),
                   'touched_split_file': save_spy.called or load_spy.called}, join(folder, f'result_{rank}.json'))
    finally:
        dist.destroy_process_group()


class TestDoSplit(unittest.TestCase):
    def test_creates_split_file_without_leftovers(self):
        with TemporaryDirectory() as tmp:
            tr_keys, val_keys = nnUNetTrainer.do_split(_fake_trainer(tmp, 0))
            self.assertEqual(sorted(os.listdir(tmp)), ['splits_final.json'])
            splits = load_json(join(tmp, 'splits_final.json'))
            self.assertEqual(len(splits), 5)
            self.assertEqual((tr_keys, val_keys), (splits[0]['train'], splits[0]['val']))

    def test_uses_existing_split_file(self):
        with TemporaryDirectory() as tmp:
            custom = [{'train': IDENTIFIERS[:15], 'val': IDENTIFIERS[15:]}]
            save_json(custom, join(tmp, 'splits_final.json'))
            tr_keys, val_keys = nnUNetTrainer.do_split(_fake_trainer(tmp, 0))
            self.assertEqual((tr_keys, val_keys), (IDENTIFIERS[:15], IDENTIFIERS[15:]))

    @unittest.skipUnless(dist.is_available(), 'torch.distributed is not available')
    def test_ddp_only_rank_0_touches_split_file(self):
        world_size = 2
        with TemporaryDirectory() as tmp:
            mp.spawn(_ddp_worker, args=(world_size, tmp), nprocs=world_size, join=True)
            results = [load_json(join(tmp, f'result_{r}.json')) for r in range(world_size)]
            splits = load_json(join(tmp, 'splits_final.json'))
            for r, result in enumerate(results):
                self.assertEqual(result['touched_split_file'], r == 0)
                self.assertEqual((result['tr'], result['val']), (splits[2]['train'], splits[2]['val']))
            self.assertFalse([f for f in os.listdir(tmp) if '.tmp.' in f])


if __name__ == '__main__':
    unittest.main()
