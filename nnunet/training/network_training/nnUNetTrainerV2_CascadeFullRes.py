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
from time import sleep
import matplotlib
from nnunet.configuration import default_num_threads
from nnunet.postprocessing.connected_components import determine_postprocessing
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import DataLoader3D, unpack_dataset
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.paths import network_training_output_dir
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.one_hot_encoding import to_one_hot
import shutil

from torch import nn

matplotlib.use("agg")


class nnUNetTrainerV2CascadeFullRes(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, previous_trainer="nnUNetTrainerV2", fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory,
                         batch_dice, stage, unpack_data, deterministic, fp16)
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, previous_trainer, fp16)

        if self.output_folder is not None:
            # task = self.output_folder.split("/")[-3]
            # plans_identifier = self.output_folder.split("/")[-2].split("__")[-1]
            task = os.path.normpath(self.output_folder).split(os.path.sep)[-3]
            plans_identifier = os.path.normpath(self.output_folder).split(os.path.sep)[-2].split("__")[-1]

            folder_with_segs_prev_stage = join(network_training_output_dir, "3d_lowres",
                                               task, previous_trainer + "__" + plans_identifier, "pred_next_stage")
            self.folder_with_segs_from_prev_stage = folder_with_segs_prev_stage
            # Do not put segs_prev_stage into self.output_folder as we need to unpack them for performance and we
            # don't want to do that in self.output_folder because that one is located on some network drive.
        else:
            self.folder_with_segs_from_prev_stage = None

    def do_split(self):
        super().do_split()
        for k in self.dataset:
            self.dataset[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                               k + "_segFromPrevStage.npz")
            assert isfile(self.dataset[k]['seg_from_prev_stage_file']), \
                "seg from prev stage missing: %s. " \
                "Please run all 5 folds of the 3d_lowres configuration of this " \
                "task!" % (self.dataset[k]['seg_from_prev_stage_file'])
        for k in self.dataset_val:
            self.dataset_val[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                                   k + "_segFromPrevStage.npz")
        for k in self.dataset_tr:
            self.dataset_tr[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                                  k + "_segFromPrevStage.npz")

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if not self.threeD:
            raise NotImplementedError("2D has no cascade")

        dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                             True, oversample_foreground_percent=self.oversample_foreground_percent,
                             pad_mode="constant", pad_sides=self.pad_all_sides)
        dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, True,
                              oversample_foreground_percent=self.oversample_foreground_percent,
                              pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

    def process_plans(self, plans):
        super().process_plans(plans)
        self.num_input_channels += (self.num_classes - 1)  # for seg from prev stage

    def setup_DA_params(self):
        super().setup_DA_params()

        self.data_aug_params["num_cached_per_thread"] = 2

        self.data_aug_params['move_last_seg_chanel_to_data'] = True
        self.data_aug_params['cascade_do_cascade_augmentations'] = True

        self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
        self.data_aug_params['cascade_random_binary_transform_p_per_label'] = 1
        self.data_aug_params['cascade_random_binary_transform_size'] = (1, 8)

        self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
        self.data_aug_params['cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
        self.data_aug_params['cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0

        # we have 2 channels now because the segmentation from the previous stage is stored in 'seg' as well until it
        # is moved to 'data' at the end
        self.data_aug_params['selected_seg_channels'] = [0, 1]
        # needed for converting the segmentation from the previous stage to one hot
        self.data_aug_params['all_segmentation_labels'] = list(range(1, self.num_classes))

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """
        if not self.was_initialized:
            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([i < net_numpool - 1 for i in range(net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)

            if training:
                self._extracted_from_initialize_37()
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')

        self.was_initialized = True

    # TODO Rename this here and in `initialize`
    def _extracted_from_initialize_37(self):
        if not isdir(self.folder_with_segs_from_prev_stage):
            raise RuntimeError(
                "Cannot run final stage of cascade. Run corresponding 3d_lowres first and predict the "
                "segmentations for the next stage")

        self.dl_tr, self.dl_val = self.get_basic_generators()
        if self.unpack_data:
            print("unpacking dataset")
            unpack_dataset(self.folder_with_preprocessed_data)
            print("done")
        else:
            print(
                "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                "will wait all winter for your model to finish!")

        self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                            self.data_aug_params[
                                                                'patch_size_for_spatialtransform'],
                                                            self.data_aug_params,
                                                            deep_supervision_scales=self.deep_supervision_scales,
                                                            pin_memory=self.pin_memory)
        self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                               also_print_to_console=False)
        self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                               also_print_to_console=False)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"

        current_mode = self.network.training
        self.network.eval()
        # save whether network is in deep supervision mode or not
        ds = self.network.do_ds
        # disable deep supervision
        self.network.do_ds = False

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            # fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            fname = os.path.basename(properties['list_of_data_files'][0])[:-12]

            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                                (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                # concat segmentation of previous step
                seg_from_prev_stage = np.load(join(self.folder_with_segs_from_prev_stage,
                                                   k + "_segFromPrevStage.npz"))['data'][None]

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                data_for_net = np.concatenate((data[:-1], to_one_hot(seg_from_prev_stage[0], range(1, self.num_classes))))

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data_for_net,
                                                                                     do_mirroring=do_mirroring,
                                                                                     mirror_axes=mirror_axes,
                                                                                     use_sliding_window=use_sliding_window,
                                                                                     step_size=step_size,
                                                                                     use_gaussian=use_gaussian,
                                                                                     all_in_gpu=all_in_gpu,
                                                                                     mixed_precision=self.fp16)[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                softmax_fname = join(output_folder, fname + ".npz") if save_softmax else None
                """There is a problem with python process communication that prevents us from communicating objects 
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, None, None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        # task = self.dataset_directory.split("/")[-1]
        task = os.path.basename(self.dataset_directory)
        job_name = self.experiment_name
        _ = aggregate_scores(
            pred_gt_tuples,
            labels=list(range(self.num_classes)),
            json_output_file=join(output_folder, "summary.json"),
            json_name=job_name + f" val tiled {use_sliding_window}",
            json_author="Fabian",
            json_task=task,
            num_threads=default_num_threads,
        )

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print(f"Could not copy gt nifti file {f} into folder {gt_nifti_folder}")
                if e is not None:
                    raise e

        # restore network deep supervision mode
        self.network.train(current_mode)
        self.network.do_ds = ds
