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
from nnunet.training.network_training.nnUNet_variants.profiling.nnUNetTrainerV2_2epochs import nnUNetTrainerV2_5epochs
from torch.nn.utils import clip_grad_norm_

try:
    from apex import amp
except ImportError:
    amp = None


class nnUNetTrainerV2_5epochs_dummyLoad(nnUNetTrainerV2_5epochs):
    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        self.some_batch = torch.rand((self.batch_size, self.num_input_channels, *self.patch_size)).float().cuda()

        self.some_gt = [torch.round(torch.rand((self.batch_size, 1, *[int(i * j) for i, j in zip(self.patch_size, k)])) * (self.num_classes - 1)).float().cuda() for k in self.deep_supervision_scales]

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data = self.some_batch
        target = self.some_gt

        self.optimizer.zero_grad()

        output = self.network(data)

        del data
        loss = self.loss(output, target)

        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        del target

        if do_backprop:
            if not self.fp16 or amp is None or not torch.cuda.is_available():
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            _ = clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return loss.detach().cpu().numpy()