from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_SGD_fixedSchedule2(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

    def maybe_update_lr(self, epoch=None):
        """
        here we go one step, then use polyLR
        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        if 0 <= ep < 500:
            new_lr = self.initial_lr
        elif 500 <= ep < 675:
            new_lr = self.initial_lr * 0.1
        elif ep >= 675:
            new_lr = poly_lr(ep - 675, self.max_num_epochs - 675, self.initial_lr * 0.1, 0.9)
        else:
            raise RuntimeError("Really unexpected things happened, ep=%d" % ep)

        self.optimizer.param_groups[0]['lr'] = new_lr
        self.print_to_log_file("lr:", self.optimizer.param_groups[0]['lr'])
