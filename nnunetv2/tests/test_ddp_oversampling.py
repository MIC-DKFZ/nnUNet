import unittest
from types import SimpleNamespace
from unittest import mock

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _ranks(world_size: int, batch_size: int, percent: float = 0.33):
    """One trainer per rank, skipping __init__: only what _set_batch_size_and_oversample reads."""
    trainers = []
    for rank in range(world_size):
        t = object.__new__(nnUNetTrainer)
        t.configuration_manager = SimpleNamespace(batch_size=batch_size)
        t.is_ddp = world_size > 1
        t.world_size = world_size
        t.global_rank = rank
        t.oversample_foreground_percent = percent
        trainers.append(t)
    return trainers


def _split(trainers):
    with mock.patch('builtins.print'):
        for t in trainers:
            t._set_batch_size_and_oversample()


def _n_forced(batch_size: int, percent: float) -> int:
    # the data loader's rule: sample i is foreground-forced if not i < round(batch_size * (1 - percent))
    return sum(not i < round(batch_size * (1 - percent)) for i in range(batch_size))


def _forced(trainers) -> int:
    return sum(_n_forced(t.batch_size, t.oversample_foreground_percent) for t in trainers)


class TestDDPOversampling(unittest.TestCase):
    def test_split_keeps_the_batch_share(self):
        for world_size, batch_size in ((1, 2), (2, 2), (2, 16), (4, 16), (8, 16), (3, 7)):
            with self.subTest(world_size=world_size, batch_size=batch_size):
                trainers = _ranks(world_size, batch_size)
                _split(trainers)
                self.assertEqual(sum(t.batch_size for t in trainers), batch_size)
                self.assertEqual(_forced(trainers), _n_forced(batch_size, 0.33))

    def test_second_split_is_the_same(self):
        # before the fix the second call split each rank's share again: 4 instead of 5 forced samples of 16 on
        # 4 GPUs, 6 on 8 GPUs
        for world_size in (2, 4, 8):
            with self.subTest(world_size=world_size):
                trainers = _ranks(world_size, 16)
                _split(trainers)
                first = [t.oversample_foreground_percent for t in trainers]
                _split(trainers)
                self.assertEqual([t.oversample_foreground_percent for t in trainers], first)
                self.assertEqual(_forced(trainers), 5)

    def test_setting_a_new_value_then_splitting(self):
        trainers = _ranks(8, 16)
        _split(trainers)
        for t in trainers:
            t.oversample_foreground_percent = 0.5
            # until the next split, reading returns what was set
            self.assertEqual(t.oversample_foreground_percent, 0.5)
        _split(trainers)
        self.assertEqual(_forced(trainers), _n_forced(16, 0.5))

    def test_without_ddp_the_value_is_untouched(self):
        t = _ranks(1, 2, 0.33)[0]
        _split([t])
        self.assertEqual(t.oversample_foreground_percent, 0.33)
        self.assertEqual(t.batch_size, 2)


if __name__ == '__main__':
    unittest.main()
