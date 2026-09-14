import unittest

import torch

from nnunetv2.utilities.helpers import autocast_if_available, dummy_context, make_grad_scaler, uses_amp


def _has_xpu() -> bool:
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


class TestAmpHelpers(unittest.TestCase):
    def test_uses_amp_on_cuda_and_xpu_only(self):
        # Device-type policy only; constructing these devices does not need hardware.
        self.assertTrue(uses_amp(torch.device('cuda')))
        self.assertTrue(uses_amp(torch.device('xpu')))
        self.assertFalse(uses_amp(torch.device('cpu')))
        self.assertFalse(uses_amp(torch.device('mps')))

    def test_make_grad_scaler_disabled_on_cpu_and_mps(self):
        self.assertIsNone(make_grad_scaler(torch.device('cpu')))
        self.assertIsNone(make_grad_scaler(torch.device('mps')))

    def test_make_grad_scaler_matches_device_when_available(self):
        if _has_xpu():
            xpu_scaler = make_grad_scaler(torch.device('xpu'))
            self.assertIsNotNone(xpu_scaler)
            self.assertTrue(xpu_scaler.is_enabled())
            self.assertEqual(xpu_scaler._device, 'xpu')

        if torch.cuda.is_available():
            cuda_scaler = make_grad_scaler(torch.device('cuda'))
            self.assertIsNotNone(cuda_scaler)
            self.assertEqual(cuda_scaler._device, 'cuda')

    def test_autocast_if_available_is_dummy_on_cpu(self):
        self.assertTrue(isinstance(autocast_if_available(torch.device('cpu')), dummy_context))

    @unittest.skipUnless(_has_xpu(), 'XPU not available')
    def test_autocast_if_available_uses_real_autocast_on_xpu(self):
        ctx = autocast_if_available(torch.device('xpu'))
        self.assertFalse(isinstance(ctx, dummy_context))


if __name__ == '__main__':
    unittest.main()
