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


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_SGD_fixedSchedule(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, max_num_epochs=1000):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16, max_num_epochs)

    def maybe_update_lr(self, epoch=None):
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        if 0 <= ep < 500:
            new_lr = self.initial_lr
        elif 500 <= ep < 675:
            new_lr = self.initial_lr * 0.1
        elif 675 <= ep < 850:
            new_lr = self.initial_lr * 0.01
        elif ep >= 850:
            new_lr = self.initial_lr * 0.001
        else:
            raise RuntimeError("Really unexpected things happened, ep=%d" % ep)

        self.optimizer.param_groups[0]['lr'] = new_lr
        self.print_to_log_file("lr:", self.optimizer.param_groups[0]['lr'])
