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

from multiprocessing.pool import Pool
from time import time

import numpy as np
import torch
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
        self.num_val_batches_per_epoch = 0 # we dont need this because this does not evaluate on full images

    def finish_online_evaluation(self):
        pass

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0):
        """
        disable nnunet postprocessing. this would just waste computation time and does not benefit brats

        !!!We run this with use_sliding_window=False per default (see on_epoch_end). This triggers fully convolutional
        inference. THIS ONLY MAKES SENSE WHEN TRAINING ON FULL IMAGES! Make sure use_sliding_window=True when running
        with default patch size (128x128x128)!!!

        per default this does not use test time data augmentation (mirroring). The reference implementation, however,
        does. I disabled it here because this eats up a lot of computation time

        """
        validation_start = time()

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)

        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
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
            properties = load_pickle(self.dataset[k]['properties_file'])
            # fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            fname = os.path.basename(properties['list_of_data_files'][0])[:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                #print(k, data.shape)

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                     do_mirroring=do_mirroring,
                                                                                     mirror_axes=mirror_axes,
                                                                                     use_sliding_window=use_sliding_window,
                                                                                     step_size=step_size,
                                                                                     use_gaussian=use_gaussian,
                                                                                     all_in_gpu=all_in_gpu,
                                                                                     verbose=False,
                                                                                     mixed_precision=self.fp16)[1]

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
                                                           interpolation_order_z, False),
                                                          )
                                                         )
                               )

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")

        # this writes a csv file into output_folder
        evaluate_regions(output_folder, self.gt_niftis_folder, self.evaluation_regions)
        csv_file = np.loadtxt(join(output_folder, 'summary.csv'), skiprows=1, dtype=str, delimiter=',')[:, 1:]

        # these are the values that are compute with np.nanmean aggregation
        whole, core, enhancing = csv_file[-4, :].astype(float)

        # do some cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.network.train(current_mode)
        validation_end = time()
        self.print_to_log_file('Running the validation took %f seconds' % (validation_end - validation_start))
        self.print_to_log_file('(the time needed for validation is included in the total epoch time!)')

        return whole, core, enhancing

    def on_epoch_end(self):
        return_value = True

        # on epoch end is called before the epoch counter is incremented, so we need to do that here to get the correct epoch number
        if (self.epoch + 1) % self.validate_every == 0:
            whole, core, enhancing = self.validate(do_mirroring=False, use_sliding_window=True,
                                                   step_size=0.5,
                                                   save_softmax=False,
                                                   use_gaussian=True, overwrite=True,
                                                   validation_folder_name='validation_after_ep_%04.0d' % self.epoch,
                                                   debug=False, all_in_gpu=True)

            here = np.mean((whole, core, enhancing))

            self.print_to_log_file("After epoch %d: whole %0.4f core %0.4f enhancing: %0.4f" %
                                   (self.epoch, whole, core, enhancing))
            self.print_to_log_file("Mean: %0.4f" % here)

            # now we need to figure out if we are done
            fully_trained_nnunet = (0.911, 0.8739, 0.7848)
            mean_dice = np.mean(fully_trained_nnunet)
            target = 0.97 * mean_dice

            self.all_val_eval_metrics.append(here)
            self.print_to_log_file("Target mean: %0.4f" % target)

            if here >= target:
                self.print_to_log_file("I am done!")
                self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
                return_value = False # this triggers early stopping

        ret_old = super().on_epoch_end()
        # if we do not achieve the target accuracy in 1000 epochs then we need to stop the training. This is not built
        # to run longer than 1000 epochs
        if not ret_old:
            return_value = ret_old

        return return_value
