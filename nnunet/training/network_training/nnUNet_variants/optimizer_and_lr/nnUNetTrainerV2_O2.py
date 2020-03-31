#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


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
