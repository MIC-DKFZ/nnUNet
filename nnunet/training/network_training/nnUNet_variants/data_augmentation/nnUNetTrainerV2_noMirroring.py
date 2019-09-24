from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_noMirroring(nnUNetTrainerV2):
    def validate(self, do_mirroring=True, use_train_mode=False, tiled=True, step=2, save_softmax=True,
                 use_gaussian=True, compute_global_dice=True, overwrite=True, validation_folder_name='validation_raw'):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction

        :param do_mirroring:
        :param use_train_mode:
        :param tiled:
        :param step:
        :param save_softmax:
        :param use_gaussian:
        :param compute_global_dice:
        :param overwrite:
        :param validation_folder_name:
        :return:
        """
        ds = self.network.do_ds
        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                  "do_mirroring was set to False")
        do_mirroring = False
        self.network.do_ds = False
        ret = super().validate(do_mirroring, use_train_mode, tiled, step, save_softmax, use_gaussian,
                               overwrite, validation_folder_name)
        self.network.do_ds = ds
        return ret

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False

    def predict_preprocessed_data_return_softmax(self, data, do_mirroring, num_repeats, use_train_mode, batch_size,
                                                 mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian,
                                                 all_in_gpu=False):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        :param data:
        :param do_mirroring:
        :param num_repeats:
        :param use_train_mode:
        :param batch_size:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param min_size:
        :param use_gaussian:
        :return:
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                  "do_mirroring was set to False")
        do_mirroring = False
        ret = super().predict_preprocessed_data_return_softmax(data, do_mirroring, num_repeats, use_train_mode,
                                                               batch_size,
                                                               mirror_axes, tiled, tile_in_z, step, min_size,
                                                               use_gaussian, all_in_gpu)
        self.network.do_ds = ds
        return ret
