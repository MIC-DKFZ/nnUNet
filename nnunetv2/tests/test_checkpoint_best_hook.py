import os
import unittest
from unittest import mock

from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _fake_init(self, plans, configuration, fold, dataset_json, device=None):
    # stands in for nnUNetTrainer.__init__: only what on_epoch_end reads
    self.logger = MetaLogger(None, False)
    self.print_to_log_file = lambda *args, **kwargs: None
    self.is_ddp = False
    self.global_rank = 0
    self.exit_training_flag = False
    self.save_every = 50
    self.num_epochs = 1000
    self.current_epoch = 0
    self._best_ema = None
    self.output_folder = 'fold_0'
    self.plot_progress_every_s = 60
    self._last_progress_plot = None


class _OnlyFromEpoch2(nnUNetTrainer):
    def _maybe_save_checkpoint_best(self):
        if self.current_epoch >= 2:
            super()._maybe_save_checkpoint_best()


def _best_writes(trainer_class, mean_fg_dice: list) -> list:
    """Epochs at which on_epoch_end wrote checkpoint_best, for a run whose pseudo Dice is mean_fg_dice."""
    with mock.patch.object(nnUNetTrainer, '__init__', _fake_init):
        t = trainer_class({}, '3d_fullres', 0, {})
    writes = []
    t.save_checkpoint = lambda filename: writes.append((t.current_epoch, os.path.basename(filename)))
    t.logger.plot_progress_png = lambda output_folder: None
    for e, dice in enumerate(mean_fg_dice):
        assert t.current_epoch == e
        for key in ('train_losses', 'val_losses', 'epoch_start_timestamps'):
            t.logger.log(key, 0., e)
        t.logger.log('dice_per_class_or_region', [dice], e)
        t.logger.log('mean_fg_dice', dice, e)  # also logs ema_fg_dice
        t.on_epoch_end()
    return [e for e, f in writes if f == 'checkpoint_best.safetensors']


class TestCheckpointBestHook(unittest.TestCase):
    def test_stock_rule(self):
        # every improvement of the EMA pseudo Dice writes checkpoint_best: EMA 0.1, 0.12, 0.118, 0.1162, 0.1546
        self.assertEqual(_best_writes(nnUNetTrainer, [0.1, 0.3, 0.1, 0.1, 0.5]), [0, 1, 4])

    def test_on_epoch_end_goes_through_the_hook(self):
        self.assertEqual(_best_writes(_OnlyFromEpoch2, [0.1, 0.3, 0.1, 0.1, 0.5]), [2, 4])


if __name__ == '__main__':
    unittest.main()
