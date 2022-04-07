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
import os
import shutil
from _warnings import warn
from collections import OrderedDict
from multiprocessing import Pool
from time import sleep, time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, isfile, load_pickle, \
    save_json #, join
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.postprocessing.connected_components import determine_postprocessing
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.distributed import awesome_allgather_function
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from torch import nn, distributed
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange


class nnUNetTrainerV2_DDP(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None,
                 unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, fp16)
        self.init_args = (
            plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice, stage, unpack_data,
            deterministic, distribute_batch_size, fp16)
        self.distribute_batch_size = distribute_batch_size
        np.random.seed(local_rank)
        torch.manual_seed(local_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_rank)
        self.local_rank = local_rank

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

        self.loss = None
        self.ce_loss = RobustCrossEntropyLoss()

        self.global_batch_size = None  # we need to know this to properly steer oversample

    def set_batch_size_and_oversample(self):
        batch_sizes = []
        oversample_percents = []

        world_size = dist.get_world_size()
        my_rank = dist.get_rank()

        if self.distribute_batch_size:
            self.global_batch_size = self.batch_size
        else:
            self.global_batch_size = self.batch_size * world_size

        batch_size_per_GPU = np.ceil(self.batch_size / world_size).astype(int)

        for rank in range(world_size):
            if self.distribute_batch_size:
                if (rank + 1) * batch_size_per_GPU > self.batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - self.batch_size)
                else:
                    batch_size = batch_size_per_GPU
            else:
                batch_size = self.batch_size

            batch_sizes.append(batch_size)

            sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
            sample_id_high = np.sum(batch_sizes)

            if sample_id_high / self.global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percents.append(0.0)
            elif sample_id_low / self.global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percents.append(1.0)
            else:
                percent_covered_by_this_rank = sample_id_high / self.global_batch_size - sample_id_low / self.global_batch_size
                oversample_percent_here = 1 - (((1 - self.oversample_foreground_percent) -
                                                sample_id_low / self.global_batch_size) / percent_covered_by_this_rank)
                oversample_percents.append(oversample_percent_here)

        print("worker", my_rank, "oversample", oversample_percents[my_rank])
        print("worker", my_rank, "batch_size", batch_sizes[my_rank])

        self.batch_size = batch_sizes[my_rank]
        self.oversample_foreground_percent = oversample_percents[my_rank]

    def save_checkpoint(self, fname, save_optimizer=True):
        if self.local_rank == 0:
            super().save_checkpoint(fname, save_optimizer)

    def plot_progress(self):
        if self.local_rank == 0:
            super().plot_progress()

    def print_to_log_file(self, *args, also_print_to_console=True):
        if self.local_rank == 0:
            super().print_to_log_file(*args, also_print_to_console=also_print_to_console)

    def process_plans(self, plans):
        super().process_plans(plans)
        self.set_batch_size_and_oversample()

    def initialize(self, training=True, force_load_plans=False):
        """
        :param training:
        :return:
        """
        if not self.was_initialized:
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
                    if self.local_rank == 0:
                        print("unpacking dataset")
                        unpack_dataset(self.folder_with_preprocessed_data)
                        print("done")
                    distributed.barrier()
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # setting weights for deep supervision losses
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
                weights[~mask] = 0
                weights = weights / weights.sum()
                self.ds_loss_weights = weights

                seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
                seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
                print("seeds train", seeds_train)
                print("seeds_val", seeds_val)
                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    seeds_train=seeds_train,
                                                                    seeds_val=seeds_val,
                                                                    pin_memory=self.pin_memory)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.network = DDP(self.network, device_ids=[self.local_rank])

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=None)
            target = to_cuda(target, gpu_id=None)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.compute_loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.compute_loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def compute_loss(self, output, target):
        total_loss = None
        for i in range(len(output)):
            # Starting here it gets spicy!
            axes = tuple(range(2, len(output[i].size())))

            # network does not do softmax. We need to do softmax for dice
            output_softmax = softmax_helper(output[i])

            # get the tp, fp and fn terms we need
            tp, fp, fn, _ = get_tp_fp_fn_tn(output_softmax, target[i], axes, mask=None)
            # for dice, compute nominator and denominator so that we have to accumulate only 2 instead of 3 variables
            # do_bg=False in nnUNetTrainer -> [:, 1:]
            nominator = 2 * tp[:, 1:]
            denominator = 2 * tp[:, 1:] + fp[:, 1:] + fn[:, 1:]

            if self.batch_dice:
                # for DDP we need to gather all nominator and denominator terms from all GPUS to do proper batch dice
                nominator = awesome_allgather_function.apply(nominator)
                denominator = awesome_allgather_function.apply(denominator)
                nominator = nominator.sum(0)
                denominator = denominator.sum(0)
            else:
                pass

            ce_loss = self.ce_loss(output[i], target[i][:, 0].long())

            # we smooth by 1e-5 to penalize false positives if tp is 0
            dice_loss = (- (nominator + 1e-5) / (denominator + 1e-5)).mean()
            if total_loss is None:
                total_loss = self.ds_loss_weights[i] * (ce_loss + dice_loss)
            else:
                total_loss += self.ds_loss_weights[i] * (ce_loss + dice_loss)
        return total_loss

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output[0].shape[1]
            output_seg = output[0].argmax(1)
            target = target[0][:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            # tp_hard, fp_hard, fn_hard = get_tp_fp_fn((output_softmax > (1 / num_classes)).float(), target,
            #                                         axes, None)
            # print_if_rank0("before allgather", tp_hard.shape)
            tp_hard = tp_hard.sum(0, keepdim=False)[None]
            fp_hard = fp_hard.sum(0, keepdim=False)[None]
            fn_hard = fn_hard.sum(0, keepdim=False)[None]

            tp_hard = awesome_allgather_function.apply(tp_hard)
            fp_hard = awesome_allgather_function.apply(fp_hard)
            fn_hard = awesome_allgather_function.apply(fn_hard)

        tp_hard = tp_hard.detach().cpu().numpy().sum(0)
        fp_hard = fp_hard.detach().cpu().numpy().sum(0)
        fn_hard = fn_hard.detach().cpu().numpy().sum(0)
        self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        self.online_eval_tp.append(list(tp_hard))
        self.online_eval_fp.append(list(fp_hard))
        self.online_eval_fn.append(list(fn_hard))

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        if self.local_rank == 0:
            self.save_debug_information()

        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = True

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))

        if self.local_rank == 0:
            # now we can delete latest as it will be identical with final
            if isfile(join(self.output_folder, "model_latest.model")):
                os.remove(join(self.output_folder, "model_latest.model"))
            if isfile(join(self.output_folder, "model_latest.model.pkl")):
                os.remove(join(self.output_folder, "model_latest.model.pkl"))

        net.do_ds = ds

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = False

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

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
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError(
                    "We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        all_keys = list(self.dataset_val.keys())
        my_keys = all_keys[self.local_rank::dist.get_world_size()]
        # we cannot simply iterate over all_keys because we need to know pred_gt_tuples and valid_labels of all cases
        # for evaluation (which is done by local rank 0)
        for k in my_keys:
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])
            if k in my_keys:
                if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                        (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                    data = np.load(self.dataset[k]['data_file'])['data']

                    print(k, data.shape)
                    data[-1][data[-1] == -1] = 0

                    softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                         do_mirroring=do_mirroring,
                                                                                         mirror_axes=mirror_axes,
                                                                                         use_sliding_window=use_sliding_window,
                                                                                         step_size=step_size,
                                                                                         use_gaussian=use_gaussian,
                                                                                         all_in_gpu=all_in_gpu,
                                                                                         mixed_precision=self.fp16)[1]

                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    """There is a problem with python process communication that prevents us from communicating obejcts
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(softmax_pred.shape, dtype=np.double) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), softmax_pred)
                        softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                             ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                               properties, interpolation_order,
                                                               self.regions_class_order,
                                                               None, None,
                                                               softmax_fname, None, force_separate_z,
                                                               interpolation_order_z),
                                                              )
                                                             )
                                   )

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        distributed.barrier()

        if self.local_rank == 0:
            # evaluate raw predictions
            self.print_to_log_file("evaluation of raw predictions")
            task = self.dataset_directory.split("/")[-1]
            job_name = self.experiment_name
            _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                                 json_output_file=join(output_folder, "summary.json"),
                                 json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                                 json_author="Fabian",
                                 json_task=task, num_threads=default_num_threads)

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
                    print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                    if e is not None:
                        raise e

        self.network.train(current_mode)
        net.do_ds = ds

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel, DDP))
        assert isinstance(self.network, tuple(valid))
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = False
        ret = net.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                             use_sliding_window=use_sliding_window, step_size=step_size,
                             patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                             use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                             pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                             mixed_precision=mixed_precision)
        net.do_ds = ds
        return ret

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                print("duh")
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]
