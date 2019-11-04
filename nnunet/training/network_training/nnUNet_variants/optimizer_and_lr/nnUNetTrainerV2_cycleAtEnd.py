from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import matplotlib.pyplot as plt


def cycle_lr(current_epoch, cycle_length=100, min_lr=1e-6, max_lr=1e-3):
    num_rising = cycle_length // 2
    epoch = current_epoch % cycle_length
    if epoch < num_rising:
        lr = min_lr + (max_lr - min_lr) / cycle_length * epoch
    else:
        lr = max_lr - (max_lr - min_lr) / cycle_length * epoch
    return lr


def plot_cycle_lr():
    xvals = list(range(1000))
    yvals = [cycle_lr(i, 100, 1e-6, 1e-3) for i in xvals]
    plt.plot(xvals, yvals)
    plt.show()
    plt.savefig("/home/fabian/temp.png")
    plt.close()


class nnUNetTrainerV2_cycleAtEnd(nnUNetTrainerV2):
    """
    after 1000 epoch, run one iteration through the cycle lr schedule. I want to see if the train loss starts
    increasing again
    """
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1100

    def maybe_update_lr(self, epoch=None):
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        if ep < 1000:
            return super().maybe_update_lr(epoch)
        else:
            new_lr = cycle_lr(ep, 100, min_lr=1e-6, max_lr=1e-3) # we don't go all the way back up to initial lr
            self.optimizer.param_groups[0]['lr'] = new_lr
            self.print_to_log_file("lr:", new_lr)


class nnUNetTrainerV2_cycleAtEnd2(nnUNetTrainerV2):
    """
    after 1000 epoch, run one iteration through the cycle lr schedule. I want to see if the train loss starts
    increasing again
    """
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1200

    def maybe_update_lr(self, epoch=None):
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        if ep < 1000:
            return super().maybe_update_lr(epoch)
        else:
            new_lr = cycle_lr(ep, 200, min_lr=1e-6, max_lr=1e-2) # we don't go all the way back up to initial lr
            self.optimizer.param_groups[0]['lr'] = new_lr
            self.print_to_log_file("lr:", new_lr)
