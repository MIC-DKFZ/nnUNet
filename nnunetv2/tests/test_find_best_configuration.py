import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from batchgenerators.utilities.file_and_folder_operations import save_json

from nnunetv2.evaluation.find_best_configuration import generate_inference_command


def _write_cascade_plans(base_dir: Path, configs=('3d_lowres', '3d_fullres', '3d_cascade_fullres')):
    plans = {
        'dataset_name': 'Dataset001_Test',
        'plans_name': 'nnUNetPlans',
        'configurations': {
            '3d_lowres': {'data_identifier': 'nnUNetPlans_3d_lowres', 'architecture': {}},
            '3d_fullres': {'data_identifier': 'nnUNetPlans_3d_fullres', 'architecture': {}},
            '3d_cascade_fullres': {
                'data_identifier': 'nnUNetPlans_3d_fullres',
                'previous_stage': '3d_lowres',
                'architecture': {},
            },
        },
    }
    for cfg in configs:
        folder = base_dir / f'nnUNetTrainer__nnUNetPlans__{cfg}'
        folder.mkdir(parents=True, exist_ok=True)
        save_json(plans, str(folder / 'plans.json'))


class TestGenerateInferenceCommand(unittest.TestCase):
    def test_cascade_generates_previous_stage_and_flag(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_cascade_plans(base)
            with patch(
                'nnunetv2.evaluation.find_best_configuration.get_output_folder',
                side_effect=lambda ds, tr, pl, cfg, fold=None: str(base / f'{tr}__{pl}__{cfg}'),
            ):
                cmd = generate_inference_command(
                    'Dataset001_Test',
                    '3d_cascade_fullres',
                    input_folder='MY_INPUT',
                    output_folder='MY_OUTPUT',
                    save_npz=True,
                )

            lines = cmd.splitlines()
            self.assertEqual(len(lines), 2, f'Expected 2 commands for cascade, got:\n{cmd}')
            self.assertIn('-c 3d_lowres', lines[0])
            self.assertIn('-i MY_INPUT', lines[0])
            self.assertIn('-o OUTPUT_FOLDER_PREV_STAGE', lines[0])
            self.assertNotIn('--save_probabilities', lines[0])
            self.assertIn('-c 3d_cascade_fullres', lines[1])
            self.assertIn('-i MY_INPUT', lines[1])
            self.assertIn('-o MY_OUTPUT', lines[1])
            self.assertIn('-prev_stage_predictions OUTPUT_FOLDER_PREV_STAGE', lines[1])
            self.assertIn('--save_probabilities', lines[1])

    def test_cascade_respects_explicit_prev_stage_folder(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_cascade_plans(base)
            with patch(
                'nnunetv2.evaluation.find_best_configuration.get_output_folder',
                side_effect=lambda ds, tr, pl, cfg, fold=None: str(base / f'{tr}__{pl}__{cfg}'),
            ):
                cmd = generate_inference_command(
                    'Dataset001_Test',
                    '3d_cascade_fullres',
                    folder_with_segs_from_prev_stage='EXISTING_PREV',
                )

            lines = cmd.splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn('-c 3d_cascade_fullres', lines[0])
            self.assertIn('-prev_stage_predictions EXISTING_PREV', lines[0])

    def test_non_cascade_generates_single_command(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_cascade_plans(base)
            with patch(
                'nnunetv2.evaluation.find_best_configuration.get_output_folder',
                side_effect=lambda ds, tr, pl, cfg, fold=None: str(base / f'{tr}__{pl}__{cfg}'),
            ):
                cmd = generate_inference_command('Dataset001_Test', '3d_fullres')

            lines = cmd.splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn('-c 3d_fullres', lines[0])
            self.assertNotIn('-prev_stage_predictions', lines[0])


if __name__ == '__main__':
    unittest.main()
