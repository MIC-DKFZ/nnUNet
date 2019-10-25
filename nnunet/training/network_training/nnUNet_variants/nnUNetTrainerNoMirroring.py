from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerNoMirroring(nnUNetTrainer):
    def validate(self, do_mirroring=True, use_train_mode=False, tiled=True, step=2, save_softmax=True,
                 use_gaussian=True, compute_global_dice=True, overwrite=True, validation_folder_name='validation_raw'):
        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                  "do_mirroring was set to False")
        do_mirroring = False
        return super().validate(do_mirroring, use_train_mode, tiled, step, save_softmax, use_gaussian,
                                compute_global_dice, overwrite, validation_folder_name)

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False
        # you can also use self.data_aug_params["mirror_axes"] to set axes for mirroring.
        # Default is self.data_aug_params["mirror_axes"] = (0, 1, 2)
        # 0, 1, 2 are the first, second and thirs spatial axes.

    def predict_preprocessed_data_return_softmax(self, data, do_mirroring, num_repeats, use_train_mode, batch_size,
                                                 mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian,
                                                 all_in_gpu=False):

        if do_mirroring:
            print("WARNING! do_mirroring was True but we cannot do that because we trained without mirroring. "
                  "do_mirroring was set to False")
        return super().predict_preprocessed_data_return_softmax(data, do_mirroring, num_repeats, use_train_mode,
                                                                batch_size, mirror_axes, tiled, tile_in_z, step,
                                                                min_size, use_gaussian)
