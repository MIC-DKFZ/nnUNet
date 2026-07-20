import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from nnunetv2.run.run_training import maybe_load_checkpoint, run_training


class DummyTrainer:
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.disable_checkpointing = False
        self.events = []

    def load_checkpoint(self, checkpoint_file: str):
        self.events.append(f"load:{os.path.basename(checkpoint_file)}")

    def run_training(self):
        self.events.append("train")

    def perform_actual_validation(self, export_probabilities: bool):
        self.events.append(f"validate:{export_probabilities}")


class TestValidationCheckpointLoading(unittest.TestCase):
    def test_validation_only_loads_final_checkpoint(self):
        with TemporaryDirectory() as output_folder:
            open(os.path.join(output_folder, 'checkpoint_final.pth'), 'a').close()
            trainer = DummyTrainer(output_folder)

            maybe_load_checkpoint(trainer, False, True)

            self.assertEqual(trainer.events, ['load:checkpoint_final.pth'])

    def test_validation_only_with_best_loads_best_without_final(self):
        with TemporaryDirectory() as output_folder:
            open(os.path.join(output_folder, 'checkpoint_best.pth'), 'a').close()
            trainer = DummyTrainer(output_folder)

            maybe_load_checkpoint(trainer, False, True, validation_with_best=True)

            self.assertEqual(trainer.events, ['load:checkpoint_best.pth'])

    def test_validation_only_with_best_reports_missing_best_checkpoint(self):
        with TemporaryDirectory() as output_folder:
            trainer = DummyTrainer(output_folder)

            with self.assertRaisesRegex(RuntimeError, 'checkpoint_best.pth is missing'):
                maybe_load_checkpoint(trainer, False, True, validation_with_best=True)

    def test_run_validation_only_with_best_loads_best_once(self):
        with TemporaryDirectory() as output_folder:
            open(os.path.join(output_folder, 'checkpoint_best.pth'), 'a').close()
            trainer = DummyTrainer(output_folder)

            with patch('nnunetv2.run.run_training.get_trainer_from_args', return_value=trainer), \
                    patch('nnunetv2.run.run_training.torch.cuda.is_available', return_value=False):
                run_training(
                    dataset_name_or_id='999',
                    configuration='3d_fullres',
                    fold=0,
                    plans_identifier='TestPlans',
                    only_run_validation=True,
                    val_with_best=True,
                    device=torch.device('cpu'),
                )

            self.assertEqual(
                trainer.events,
                ['load:checkpoint_best.pth', 'validate:False'],
            )

    def test_training_with_val_best_still_loads_best_after_training(self):
        with TemporaryDirectory() as output_folder:
            trainer = DummyTrainer(output_folder)

            with patch('nnunetv2.run.run_training.get_trainer_from_args', return_value=trainer), \
                    patch('nnunetv2.run.run_training.torch.cuda.is_available', return_value=False):
                run_training(
                    dataset_name_or_id='999',
                    configuration='3d_fullres',
                    fold=0,
                    plans_identifier='TestPlans',
                    only_run_validation=False,
                    val_with_best=True,
                    device=torch.device('cpu'),
                )

            self.assertEqual(
                trainer.events,
                ['train', 'load:checkpoint_best.pth', 'validate:False'],
            )


if __name__ == '__main__':
    unittest.main()
