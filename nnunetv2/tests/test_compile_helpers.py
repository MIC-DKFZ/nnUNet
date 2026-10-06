import os
import unittest
from unittest import mock

import torch

from nnunetv2.utilities.helpers import is_optimized_module, set_default_inductor_cache_dir


class TestCompileHelpers(unittest.TestCase):
    def test_inductor_cache_dir(self):
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir
        base = {k: v for k, v in os.environ.items() if k not in ('TORCHINDUCTOR_CACHE_DIR', 'XDG_CACHE_HOME')}
        expected = os.path.join('/x', 'nnunetv2', 'torchinductor')
        # unset or inductor's own /tmp default -> persistent default; anything else is the user's choice and stays
        for preset, result in ((None, expected), (default_cache_dir(), expected), ('/my/cache', '/my/cache')):
            env = dict(base, XDG_CACHE_HOME='/x')
            if preset is not None:
                env['TORCHINDUCTOR_CACHE_DIR'] = preset
            with mock.patch.dict(os.environ, env, clear=True):
                set_default_inductor_cache_dir()
                self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'], result, preset)

    def test_is_optimized_module(self):
        net = torch.nn.Linear(2, 2)
        self.assertFalse(is_optimized_module(net))
        self.assertTrue(is_optimized_module(torch.compile(net)))


if __name__ == '__main__':
    unittest.main()
