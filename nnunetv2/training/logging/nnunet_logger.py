import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any


class MetaLogger(object):
    """A meta logger that bundles multiple loggers behind a single interface.

    The default configuration includes a local logger used for reading values,
    plotting progress, and checkpointing.
    """

    def __init__(self, verbose: bool = False):
        """Initialize the meta logger.

        Args:
            verbose: If True, enables verbose logging in the local logger.
        """
        self.loggers = {}
        local_logger = LocalLogger(verbose)
        self.local_logger_key = local_logger.get_logger_key()
        self.loggers[self.local_logger_key] = local_logger

    def update_config(self, config: dict):
        """Add a new or update an existing experiment configuration to the logger.

        Args:
            config: Logger configuration options.
        """
        for logger in self.loggers.values():
            logger.update_config(config)

    def log(self, key: str, value: Any, step: int):
        """Log a value for a given step.

        Args:
            key: Metric or field name.
            value: Value to log.
            step: Step index (typically epoch).
        """
        for logger in self.loggers.values():
            logger.log(key, value, step)

    def log_summary(self, key: str, value: Any):
        """Log a summary value. These are usually values that are not logged every step but only once. 
        This can be for example the final validation Dice.

        Args:
            key: Metric or field name.
            value: Value to summarize.
        """
        for logger in self.loggers.values():
            logger.log_summary(key, value)

    def get_value(self, key: str, step: Any):
        """Fetch a logged value from the local logger.

        Args:
            key: Metric or field name.
            step: Step index to retrieve, or None to return all values.

        Returns:
            The logged value or list of values from the local logger.
        """
        return self.loggers[self.local_logger_key].get_value(key, step)

    def plot_progress_png(self, output_folder: str):
        """Write a progress plot PNG using local logger data.

        Args:
            output_folder: Directory where the plot image is saved.
        """
        self.loggers[self.local_logger_key].plot_progress_png(output_folder)

    def get_checkpoint(self):
        """Return the local logger checkpoint data.

        Returns:
            The checkpoint payload used to restore logging state.
        """
        return self.loggers[self.local_logger_key].get_checkpoint()

    def load_checkpoint(self, checkpoint: dict):
        """Restore the local logger from a checkpoint payload.

        Args:
            checkpoint: Checkpoint data returned by `get_checkpoint`.
        """
        self.loggers[self.local_logger_key].load_checkpoint(checkpoint)


class LocalLogger:
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        self.verbose = verbose
        # shut up, this logging is great

    def get_logger_key(self):
        return "local"
    
    def update_config(self, config: dict):
        pass

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

    def log_summary(self, key, value):
        pass

    def get_value(self, key, step):
        if step is not None:
            return self.my_fantastic_logging[key][step]
        else:
            return self.my_fantastic_logging[key]

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
