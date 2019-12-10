import torch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
try:
    from apex import amp
except ImportError:
    amp = None


class nnUNetTrainerV2_O2(nnUNetTrainerV2):
    """
    force O2 in amp
    """
    def _maybe_init_amp(self):
        if self.fp16:
            if not self.amp_initialized:
                if amp is not None:
                    self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level="O1")
                    self.amp_initialized = True
                else:
                    raise RuntimeError("WARNING: FP16 training was requested but nvidia apex is not installed. "
                                       "Install it from https://github.com/NVIDIA/apex")
