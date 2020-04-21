from multiprocessing.pool import Pool
import numpy as np
from nnunet.configuration import default_num_threads
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_brats_regions


class nnUNetTrainerV2_fullEvals(nnUNetTrainerV2):
    """
    this trainer only works for brats and nothing else
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.validate_every = 1
        self.evaluation_regions = get_brats_regions()
        self.num_val_batches_per_epoch = 0
        self.num_batches_per_epoch = 10

    def finish_online_evaluation(self):
        pass

    def validate(self, do_mirroring: bool = True, use_train_mode: bool = False, tiled: bool = True, step: int = 2,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z: int = 0):
        """
        disable nnunet postprocessing. this would just waste computation time and does not benefit brats

        per default this does not use test time data augmentation (mirroring). The reference implementation, however,
        does.
        I disabled it here because this eats up a lot of computation time
        """
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)

        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_train_mode': use_train_mode,
                         'tiled': tiled,
                         'step': step,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'force_separate_z': force_separate_z,
                         'interpolation_order': interpolation_order,
                         'interpolation_order_z': interpolation_order_z,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        export_pool = Pool(default_num_threads)
        results = []

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                print(k, data.shape)

                softmax_pred = self.predict_preprocessed_data_return_softmax(data[:-1], do_mirroring, 1,
                                                                             use_train_mode, 1, mirror_axes, tiled,
                                                                             True, step, self.patch_size,
                                                                             use_gaussian=use_gaussian,
                                                                             all_in_gpu=all_in_gpu)

                # this does not do anything in brats -> remove this line
                # softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, None, None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        evaluate_regions(output_folder, self.gt_niftis_folder, self.evaluation_regions)

        # this writes a csv file into output_folder
        import IPython;
        IPython.embed()
        csv_file = np.loadtxt(join(output_folder, 'summary.csv'), skiprows=1, dtype=str)[:, 1:]
        # these are the values that are compute with np.nanmean aggregation
        whole, core, enhancing = csv_file[-3, :].astype(float)
        return whole, core, enhancing

    def on_epoch_end(self):
        ret = super().on_epoch_end()

        # on epoch end is called before the epoch counter is incremented, so we need to do that here to get the correct epoch number
        if (self.epoch + 1) % 5 == self.validate_every:
            whole, core, enhancing = self.validate(do_mirroring=True, use_train_mode=False, tiled=False, step=2,
                                                   save_softmax=False,
                                                   use_gaussian=True, overwrite=True,
                                                   validation_folder_name='validation_after_ep_%04.0d' % (self.epoch),
                                                   debug=False, all_in_gpu=False)

            here = np.mean((whole, core, enhancing))

            self.print_to_log_file("After epoch %d: whole %0.4f core %0.4f enhancing: %0.4f" %
                                   (self.epoch, whole, core, enhancing))
            self.print_to_log_file("Mean: %0.4f" % here)

            # now we need to figure out if we are done
            fully_trained_nnunet = (0.911, 0.8739, 0.7848)
            mean_dice = np.mean(fully_trained_nnunet)
            target = 0.97 * mean_dice

            if here >= target:
                self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
                self.epoch = self.max_num_epochs # this will then cause the training to abort
        return ret
