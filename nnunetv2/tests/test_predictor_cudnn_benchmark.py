import unittest

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class TestPredictorLeavesCudnnBenchmarkAlone(unittest.TestCase):
    def test_setting_is_left_as_it_is(self):
        before = torch.backends.cudnn.benchmark
        try:
            for value in (False, True):
                torch.backends.cudnn.benchmark = value
                # creating the predictor does not need a GPU
                nnUNetPredictor(device=torch.device('cuda', 0))
                self.assertEqual(torch.backends.cudnn.benchmark, value)
        finally:
            torch.backends.cudnn.benchmark = before


if __name__ == '__main__':
    unittest.main()
