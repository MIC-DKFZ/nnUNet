from batchgenerators.utilities.file_and_folder_operations import join
from pytorch_lightning import Callback
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetProgressPngCallback(Callback):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        sns.set()
        my_log = pl_module.my_fantastic_logging

        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(20, 18 * 2))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(pl_module.current_epoch + 1))
        ax.plot(x_values, my_log['train_losses'], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, my_log['val_losses'], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, my_log['mean_fg_dice'], color='g', ls='--', label="pseudo dice", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent
        ax = ax_all[1]
        x_values = list(range(pl_module.current_epoch + 1))
        ax.plot(x_values, [i - j for i, j in zip(my_log['epoch_end_timestamps'], my_log['epoch_start_timestamps'])], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        x_values = list(range(pl_module.current_epoch + 1))
        ax.plot(x_values, my_log['lrs'], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(pl_module.output_folder, "progress.png"))
        plt.close()
