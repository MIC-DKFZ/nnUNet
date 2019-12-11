from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_resample33(nnUNetTrainerV2):
    def validate(self, do_mirroring: bool = True, use_train_mode: bool = False, tiled: bool = True, step: int = 2,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0):
        return super().validate(do_mirroring, use_train_mode, tiled, step, save_softmax, use_gaussian,
                                overwrite, validation_folder_name, debug, all_in_gpu,
                                force_separate_z=False, interpolation_order=3,
                                interpolation_order_z=3)

    def preprocess_predict_nifti(self, input_files, output_file=None, softmax_ouput_file=None):
        """
        Use this to predict new data
        :param input_files:
        :param output_file:
        :param softmax_ouput_file:
        :return:
        """
        print("preprocessing...")
        d, s, properties = self.preprocess_patient(input_files)
        print("predicting...")
        pred = self.predict_preprocessed_data_return_softmax(d, self.data_aug_params["do_mirror"], 1, False, 1,
                                                             self.data_aug_params['mirror_axes'], True, True, 2,
                                                             self.patch_size, True)
        pred = pred.transpose([0] + [i + 1 for i in self.transpose_backward])

        print("resampling to original spacing and nifti export...")
        save_segmentation_nifti_from_softmax(pred, output_file, properties, 3, None, None, None, softmax_ouput_file,
                                             None, force_separate_z=False, interpolation_order_z=3)
        print("done")
