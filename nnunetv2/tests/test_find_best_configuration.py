import unittest

from nnunetv2.evaluation.find_best_configuration import find_best_configuration


class TestFindBestConfiguration(unittest.TestCase):
    def test_folds_none_is_rejected_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "folds must be a list or tuple"):
            find_best_configuration("Dataset001_Test", allowed_trained_models=(), folds=None)


if __name__ == "__main__":
    unittest.main()
