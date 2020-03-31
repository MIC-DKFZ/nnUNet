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


from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes


class nnUNetTrainerV2CascadeFullRes_noConnComp(nnUNetTrainerV2CascadeFullRes):
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 8)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.0
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0


class nnUNetTrainerV2CascadeFullRes_smallerBinStrel(nnUNetTrainerV2CascadeFullRes):
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 5)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0


class nnUNetTrainerV2CascadeFullRes_EducatedGuess(nnUNetTrainerV2CascadeFullRes):
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.5
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 0.5
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 5)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.10
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0


class nnUNetTrainerV2CascadeFullRes_EducatedGuess2(nnUNetTrainerV2CascadeFullRes):
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.5
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 0.5
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 5)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.0
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.10
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0


class nnUNetTrainerV2CascadeFullRes_EducatedGuess3(nnUNetTrainerV2CascadeFullRes):
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 1
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 0.33
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 5)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.0
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.10
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0

