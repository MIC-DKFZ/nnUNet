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

from collections import OrderedDict
from time import sleep

import numpy as np
import torch
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join, subfiles, isfile
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.distributed import awesome_allgather_function
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler


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
        torch.cuda.manual_seed_all(local_rank)
        self.local_rank = local_rank

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

        self.val_loss_ma_alpha = 0.95
        self.val_loss_MA = None

        self.loss = None
        self.ce_loss = CrossentropyND()

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

    def initialize_network(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """
        self.print_to_log_file(self.net_num_pool_op_kernel_sizes)
        self.print_to_log_file(self.net_conv_kernel_sizes)

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def process_plans(self, plans):
        super().process_plans(plans)
        self.set_batch_size_and_oversample()

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
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
                    else:
                        # we need to wait until worker 0 has finished unpacking
                        npz_files = subfiles(self.folder_with_preprocessed_data, suffix=".npz", join=False)
                        case_ids = [i[:-4] for i in npz_files]
                        all_present = all(
                            [isfile(join(self.folder_with_preprocessed_data, i + ".npy")) for i in case_ids])
                        while not all_present:
                            print("worker", self.local_rank, "is waiting for unpacking")
                            sleep(3)
                            all_present = all(
                                [isfile(join(self.folder_with_preprocessed_data, i + ".npy")) for i in case_ids])
                        # there is some slight chance that there may arise some error because dataloader are loading a file
                        # that is still being written by worker 0. We ignore this for now an address it only if it becomes
                        # relevant
                        # (this can occur because while worker 0 writes the file is technically present so the other workers
                        # will proceed and eventually try to read it)
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
                                                                    seeds_val=seeds_val)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self._maybe_init_amp()
            self.network = DDP(self.network)

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        data = to_cuda(data, gpu_id=None)
        target = to_cuda(target, gpu_id=None)

        self.optimizer.zero_grad()

        output = self.network(data)
        del data

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

            ce_loss = self.ce_loss(output[i], target[i])

            # we smooth by 1e-5 to penalize false positives if tp is 0
            dice_loss = (- (nominator + 1e-5) / (denominator + 1e-5)).mean()
            if total_loss is None:
                total_loss = self.ds_loss_weights[i] * (ce_loss + dice_loss)
            else:
                total_loss += self.ds_loss_weights[i] * (ce_loss + dice_loss)

        if run_online_evaluation:
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
                # print_if_rank0("after allgather", tp_hard.shape)

                # print_if_rank0("after sum", tp_hard.shape)

                self.run_online_evaluation(tp_hard.detach().cpu().numpy().sum(0),
                                           fp_hard.detach().cpu().numpy().sum(0),
                                           fn_hard.detach().cpu().numpy().sum(0))
        del target

        if do_backprop:
            if not self.fp16 or amp is None:
                total_loss.backward()
            else:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            _ = clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return total_loss.detach().cpu().numpy()

    def run_online_evaluation(self, tp, fp, fn):
        self.online_eval_foreground_dc.append(list((2 * tp) / (2 * tp + fp + fn + 1e-8)))
        self.online_eval_tp.append(list(tp))
        self.online_eval_fp.append(list(fp))
        self.online_eval_fn.append(list(fn))

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = True
        ret = nnUNetTrainer.run_training(self)
        net.do_ds = ds
        return ret

    def validate(self, do_mirroring: bool = True, use_train_mode: bool = False, tiled: bool = True, step: int = 2,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0):
        if self.local_rank == 0:
            if isinstance(self.network, DDP):
                net = self.network.module
            else:
                net = self.network
            ds = net.do_ds
            net.do_ds = False
            ret = nnUNetTrainer.validate(self, do_mirroring, use_train_mode, tiled, step, save_softmax, use_gaussian,
                                         overwrite, validation_folder_name, debug, all_in_gpu,
                                         force_separate_z=force_separate_z, interpolation_order=interpolation_order,
                                         interpolation_order_z=interpolation_order_z)
            net.do_ds = ds
            return ret

    def predict_preprocessed_data_return_softmax(self, data, do_mirroring, num_repeats, use_train_mode, batch_size,
                                                 mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian,
                                                 all_in_gpu=False):
        """
        Don't use this. If you need softmax output, use preprocess_predict_nifti and set softmax_output_file.
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
        :param use_temporal:
        :return:
        """
        valid = list((SegmentationNetwork, nn.DataParallel, DDP))
        assert isinstance(self.network, tuple(valid))
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        ds = net.do_ds
        net.do_ds = False
        ret = net.predict_3D(data, do_mirroring, num_repeats, use_train_mode, batch_size, mirror_axes,
                             tiled, tile_in_z, step, min_size, use_gaussian=use_gaussian,
                             pad_border_mode=self.inference_pad_border_mode,
                             pad_kwargs=self.inference_pad_kwargs, all_in_gpu=all_in_gpu)[2]
        net.do_ds = ds
        return ret

    def load_checkpoint_ram(self, saved_model, train=True):
        """
        used for if the checkpoint is already in ram
        :param saved_model:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in saved_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                print("duh")
                key = key[7:]
            new_state_dict[key] = value

        # if we are fp16, then we need to reinitialize the network and the optimizer. Otherwise amp will throw an error
        if self.fp16:
            self.network, self.optimizer, self.lr_scheduler = None, None, None
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            # we need to reinitialize DDP here
            self.network = DDP(self.network)

        self.network.load_state_dict(new_state_dict)
        self.epoch = saved_model['epoch']
        if train:
            optimizer_state_dict = saved_model['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and saved_model[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(saved_model['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = saved_model[
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

        self.amp_initialized = False
        self._maybe_init_amp()
