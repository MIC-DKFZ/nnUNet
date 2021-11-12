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


from _warnings import warn
from typing import Tuple

import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.neural_network import SegmentationNetwork
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime
from tqdm import trange
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class NetworkTrainer(object):
    def __init__(self, deterministic=True, fp16=False):
        """
        A generic class that can train almost any neural network (RNNs excluded). It provides basic functionality such
        as the training loop, tracking of training and validation losses (and the target metric if you implement it)
        Training can be terminated early if the validation loss (or the target metric if implemented) do not improve
        anymore. This is based on a moving average (MA) of the loss/metric instead of the raw values to get more smooth
        results.

        What you need to override:
        - __init__
        - initialize
        - run_online_evaluation (optional)
        - finish_online_evaluation (optional)
        - validate
        - predict_test_case
        """
        self.fp16 = fp16
        self.amp_grad_scaler = None

        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        ################# SET THESE IN self.initialize() ###################################
        self.network: Tuple[SegmentationNetwork, nn.DataParallel] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        ################# SET THESE IN INIT ################################################
        self.output_folder = None
        self.fold = None
        self.loss = None
        self.dataset_directory = None

        ################# SET THESE IN LOAD_DATASET OR DO_SPLIT ############################
        self.dataset = None  # these can be None for inference mode
        self.dataset_tr = self.dataset_val = None  # do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        self.val_eval_criterion_MA_eps = 5e-4 # new MA must be at least this much better (greater)
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = 1000
        self.num_batches_per_epoch = 250
        self.num_val_batches_per_epoch = 50
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_MA_val_eval_criterion_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.best_epoch_based_on_MA_val_eval_criterion = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        self.use_progress_bar = False
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

        ################# Settings for saving checkpoints ##################################
        self.save_every = 50
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

    @abstractmethod
    def initialize(self, training=True):
        """
        create self.output_folder

        modify self.output_folder if you are doing cross-validation (one folder per fold)

        set self.tr_gen and self.val_gen

        call self.initialize_network and self.initialize_optimizer_and_scheduler (important!)

        finally set self.was_initialized to True
        :param training:
        :return:
        """

    @abstractmethod
    def load_dataset(self):
        pass

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            self.print_to_log_file("Creating new split...")
            splits = []
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
            save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(self.output_folder, "model_best.model")):
            self.load_checkpoint(join(self.output_folder, "model_best.model"), train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        if isfile(join(self.output_folder, "model_final_checkpoint.model")):
            return self.load_checkpoint(join(self.output_folder, "model_final_checkpoint.model"), train=train)
        if isfile(join(self.output_folder, "model_latest.model")):
            return self.load_checkpoint(join(self.output_folder, "model_latest.model"), train=train)
        if isfile(join(self.output_folder, "model_best.model")):
            return self.load_best_checkpoint(train)
        raise RuntimeError("No checkpoint found")

    def load_final_checkpoint(self, train=False):
        filename = join(self.output_folder, "model_final_checkpoint.model")
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    @abstractmethod
    def initialize_network(self):
        """
        initialize self.network here
        :return:
        """
        pass

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """
        initialize self.optimizer and self.lr_scheduler (if applicable) here
        :return:
        """
        pass

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
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
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

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

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

        self._maybe_init_amp()

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()

    def plot_network_architecture(self):
        """
        can be implemented (see nnUNetTrainer) but does not have to. Not implemented here because it imposes stronger
        assumptions on the presence of class variables
        :return:
        """
        pass

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

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
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            self.print_to_log_file("done")

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]

    def manage_patience(self):
        """
        Modified based on original code.
        The early stopping is handled based on validation metric instead of training loss
        """
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            # if self.best_MA_tr_loss_for_patience is None:
            #     self.best_MA_tr_loss_for_patience = self.train_loss_MA

            # if self.best_epoch_based_on_MA_tr_loss is None:
            #     self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_MA_val_eval_criterion_for_patience is None:
                self.best_MA_val_eval_criterion_for_patience = self.val_eval_criterion_MA

            if self.best_epoch_based_on_MA_val_eval_criterion is None:
                self.best_epoch_based_on_MA_val_eval_criterion = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the MA of the valdiation metric has improved. If yes then reset patience, else increase patience
            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA + self.val_eval_criterion_MA_eps:
                self.best_MA_val_eval_criterion_for_patience = self.val_eval_criterion_MA
                self.best_epoch_based_on_MA_val_eval_criterion = self.epoch

            # # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # # increase patience
            # if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
            #     self.best_MA_tr_loss_for_patience = self.train_loss_MA
            #     self.best_epoch_based_on_MA_tr_loss = self.epoch
            #     #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            # else:
            #     pass
            #     #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
            #     #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_val_eval_criterion > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_val_eval_criterion = self.epoch - self.patience // 2
                else:
                    self.print_to_log_file("My patience ended")
                    continue_training = False

        return continue_training

    def on_epoch_end(self):
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, *args, **kwargs):
        """
        Can be implemented, does not have to
        :param output_torch:
        :param target_npy:
        :return:
        """
        pass

    def finish_online_evaluation(self):
        """
        Can be implemented, does not have to
        :return:
        """
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    def find_lr(self, num_iters=1000, init_value=1e-6, final_value=10., beta=0.98):
        """
        stolen and adapted from here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        :param num_iters:
        :param init_value:
        :param final_value:
        :param beta:
        :return:
        """
        import math
        self._maybe_init_amp()
        mult = (final_value / init_value) ** (1 / num_iters)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        for batch_num in range(1, num_iters + 1):
            # +1 because this one here is not designed to have negative loss...
            loss = self.run_iteration(self.tr_gen, do_backprop=True, run_online_evaluation=False).data.item() + 1

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        import matplotlib.pyplot as plt
        lrs = [10 ** i for i in log_lrs]
        fig = plt.figure()
        plt.xscale('log')
        plt.plot(lrs[10:-5], losses[10:-5])
        plt.savefig(join(self.output_folder, "lr_finder.png"))
        plt.close()
        return log_lrs, losses
