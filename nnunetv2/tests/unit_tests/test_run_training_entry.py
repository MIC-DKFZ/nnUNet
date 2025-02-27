import unittest
from unittest.mock import patch
import tempfile
import os
import yaml
from io import StringIO
import sys

# Move up three levels
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent_dir)

from run.run_training import run_training_entry  # Import the function from your actual script


class TestRunTrainingEntry(unittest.TestCase):
    def setUp(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.original_argv = sys.argv.copy()  # Make a copy
        self.mock_stdout = StringIO()
        self.mock_stderr = StringIO()
        sys.stdout = self.mock_stdout
        sys.stderr = self.mock_stderr

    def tearDown(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        sys.argv = self.original_argv  # Restore the copy
        patch.stopall()
        self.mock_stdout.close()
        self.mock_stderr.close()

    def run_with_mocks(self, test_function):
        self.mock_stdout.seek(0)
        self.mock_stderr.seek(0)
        self.mock_stdout.truncate(0)
        self.mock_stderr.truncate(0)
        test_function()
        return self.mock_stdout.getvalue() + self.mock_stderr.getvalue()

    def create_temp_config(self, config_data):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_file:
            yaml.dump(config_data, temp_file)
        return temp_file.name

    def test_without_config(self):
        with patch('sys.argv', ['script_name', 'dataset_name', 'configuration', '0']):
            # Perform the test call of run_training_entry
            output = self.run_with_mocks(lambda: run_training_entry(testing=True))

        self.assertIn("Script parameters:", output)

    def test_with_valid_config(self):
        config = {
            'initial_lr': {'value': 1e-3, 'type': 'float', 'description': 'Initial learning rate'},
            'num_epochs': {'value': 100, 'type': 'int', 'description': 'Number of epochs'}
        }
        config_file = self.create_temp_config(config)
        with patch('sys.argv', ['script_name', 'dataset_name', 'configuration', '0', '--config', config_file]):
            # Perform the test call of run_training_entry
            output = self.run_with_mocks(lambda: run_training_entry(testing=True))

        self.assertIn("initial_lr: 0.001", output)
        self.assertIn("num_epochs: 100", output)
        os.unlink(config_file)

    def test_edge_case_values(self):
        config = {
            'very_small_float': {'value': 1e-20, 'type': 'float', 'description': 'Very small float'},
            'very_large_int': {'value': 1000000000, 'type': 'int', 'description': 'Very large int'}
        }
        config_file = self.create_temp_config(config)
        with patch('sys.argv', ['script_name', 'dataset_name', 'configuration', '0', '--config', config_file]):
            # Perform the test call of run_training_entry
            output = self.run_with_mocks(lambda: run_training_entry(testing=True))

        self.assertIn("very_small_float: 1e-20", output)
        self.assertIn("very_large_int: 1000000000", output)
        os.unlink(config_file)

    def test_conflicting_values(self):
        config = {
            'num_epochs': {'value': 100, 'type': 'int', 'description': 'Number of epochs'}
        }
        config_file = self.create_temp_config(config)
        with patch('sys.argv', ['script_name', 'dataset_name', 'configuration', '0', '--config', config_file, '--num_epochs', '200']):
            # Perform the test call of run_training_entry
            output = self.run_with_mocks(lambda: run_training_entry(testing=True))

        self.assertIn("num_epochs: 200", output)
        self.assertIn("num_epochs: type check passed (int)", output)
        os.unlink(config_file)

    def test_type_mismatch(self):
        config = {
            'initial_lr': {'value': 'not_a_float', 'type': 'float', 'description': 'Initial learning rate'}
        }
        config_file = self.create_temp_config(config)
        with patch('sys.argv', ['script_name', 'dataset_name', 'configuration', '0', '--config', config_file]):
            with self.assertRaises(ValueError) as context:
                run_training_entry(testing=True)

        self.assertIn("Invalid value for initial_lr: not_a_float. Expected type: float", str(context.exception))
        os.unlink(config_file)

    def test_missing_required_args(self):
        with patch('sys.argv', ['script_name']):
            with self.assertRaises(SystemExit):
                run_training_entry(testing=True)


if __name__ == '__main__':
    unittest.main()
