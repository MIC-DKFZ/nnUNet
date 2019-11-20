from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes


class nnUNetTrainerV2CascadeFullRes_noConnComp(nnUNetTrainerV2CascadeFullRes):
    def setup_DA_params(self):
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 8)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.0
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0


class nnUNetTrainerV2CascadeFullRes_smallerBinStrel(nnUNetTrainerV2CascadeFullRes):
    def setup_DA_params(self):
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 5)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0


class nnUNetTrainerV2CascadeFullRes_EducatedGuess(nnUNetTrainerV2CascadeFullRes):
    def setup_DA_params(self):
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.5
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 0.5
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 5)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.10
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0

