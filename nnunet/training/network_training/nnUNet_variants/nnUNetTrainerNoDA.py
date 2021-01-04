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


import matplotlib
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.data_augmentation_noDA import get_no_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader3D, DataLoader2D
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from torch import nn

matplotlib.use("agg")


class nnUNetTrainerNoDA(nnUNetTrainer):
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent
                                 , pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.patch_size, self.patch_size, self.batch_size,
                                 transpose=self.plans.get('transpose_forward'),
                                 oversample_foreground_percent=self.oversample_foreground_percent
                                 , pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  transpose=self.plans.get('transpose_forward'),
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        self.setup_DA_params()

        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                  "_stage%d" % self.stage)
        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                print("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                print("done")
            else:
                print("INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                      "will wait all winter for your model to finish!")
            self.tr_gen, self.val_gen = get_no_augmentation(self.dl_tr, self.dl_val, params=self.data_aug_params)
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
        else:
            pass
        self.initialize_network()
        assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True
        self.data_aug_params['mirror_axes'] = ()
