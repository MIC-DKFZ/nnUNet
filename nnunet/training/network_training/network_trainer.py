from _warnings import warn
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold
matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
from datetime import datetime
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime


try:
    from apex import amp
except ImportError:
    amp = None


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
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        self.fp16 = fp16

        if deterministic:
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        ################# SET THESE IN self.initialize() ###################################
        self.network = None
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
        self.dataset_tr = self.dataset_val = None # do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.save_every = 50
        self.save_latest_only = True
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
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = [] # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic


    @abstractmethod
    def initialize(self, training=True):
        """
        create self.output_folder

        modify self.output_folder if you are doing cross-validation (one folder per fold)

        set self.tr_gen and self.val_gen
        
        set self.network, self.optimizer and self.lr_scheduler
        
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
            if len(self.all_val_eval_metrics) == len(self.all_val_losses):
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
                                         (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second))
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
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            for key in lr_sched_state_dct.keys():
                lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        torch.save({
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics)},
            fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        self.load_checkpoint(join(self.output_folder, "model_best.model"), train=train)

    def load_latest_checkpoint(self, train=True):
        if isfile(join(self.output_folder, "model_final_checkpoint.model")):
            return self.load_checkpoint(join(self.output_folder, "model_final_checkpoint.model"), train=train)
        if isfile(join(self.output_folder, "model_latest.model")):
            return self.load_checkpoint(join(self.output_folder, "model_latest.model"), train=train)
        all_checkpoints = [i for i in os.listdir(self.output_folder) if i.endswith(".model") and i.find("_ep_") != -1]
        if len(all_checkpoints) == 0:
            return self.load_best_checkpoint(train=train)
        corresponding_epochs = [int(i.split("_")[-1].split(".")[0]) for i in all_checkpoints]
        checkpoint = all_checkpoints[np.argmax(corresponding_epochs)]
        self.load_checkpoint(join(self.output_folder, checkpoint), train=train)

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        self.load_checkpoint_ram(saved_model, train)

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
                key = key[7:]
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)
        self.epoch = saved_model['epoch']
        if train:
            optimizer_state_dict = saved_model['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.load_state_dict(saved_model['lr_scheduler_state_dict'])

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = saved_model['plot_stuff']

    def _maybe_init_amp(self):
        # we use fp16 for training only, not inference
        if self.fp16:
            if amp is not None:
                self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level="O1")
            else:
                self.print_to_log_file("WARNING: FP16 training was requested but nvidia apex is not installed. "
                                       "Install it from https://github.com/NVIDIA/apex")

    def plot_network_architecture(self):
        """
        can be implemented (see nnUNetTrainer) but does not have to. Not implemented here because it imposes stronger
        assumptions on the presence of class variables
        :return:
        """
        pass

    def run_training(self):
        torch.cuda.empty_cache()

        self._maybe_init_amp()

        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        maybe_mkdir_p(self.output_folder)

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()
            for b in range(self.num_batches_per_epoch):
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
                self.print_to_log_file("val loss (train=False): %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("val loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            epoch_end_time = time()

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()
            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))

        self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
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
        if self.epoch % self.save_every == (self.save_every - 1):
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
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.print_to_log_file("saving best epoch checkpoint...")
                self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                self.print_to_log_file(
                    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def on_epoch_end(self):
        self.finish_online_evaluation() # does not have to do anything, but can be used to update self.all_val_eval_
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

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        self.optimizer.zero_grad()

        output = self.network(data)
        del data
        l = self.loss(output, target)

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        if do_backprop:
            if not self.fp16 or amp is None:
                l.backward()
            else:
                with amp.scale_loss(l, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()

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
        mult = (final_value / init_value) ** (1/num_iters)
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
            avg_loss = beta * avg_loss + (1-beta) * loss
            smoothed_loss = avg_loss / (1 - beta**batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
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
