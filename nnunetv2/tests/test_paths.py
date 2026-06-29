import os
import unittest
from unittest.mock import patch

from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import find_candidate_datasets


class TestPaths(unittest.TestCase):
    def test_missing_env_var_only_raises_when_converted_to_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('nnUNet_raw', None)
            self.assertFalse(nnUNet_raw.is_set())
            with self.assertRaisesRegex(RuntimeError, 'nnUNet_raw is not defined'):
                os.fspath(nnUNet_raw)

    def test_env_var_is_resolved_lazily(self):
        with patch.dict(os.environ, {'nnUNet_raw': '/tmp/nnunet_raw_test'}, clear=False):
            self.assertTrue(nnUNet_raw.is_set())
            self.assertEqual(os.fspath(nnUNet_raw), '/tmp/nnunet_raw_test')

    def test_dataset_lookup_tolerates_missing_env_vars(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('nnUNet_raw', None)
            os.environ.pop('nnUNet_preprocessed', None)
            os.environ.pop('nnUNet_results', None)
            self.assertEqual(len(find_candidate_datasets(999)), 0)


if __name__ == '__main__':
    unittest.main()
