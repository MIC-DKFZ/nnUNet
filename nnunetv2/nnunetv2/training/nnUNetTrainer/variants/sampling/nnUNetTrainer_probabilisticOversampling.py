import numpy as np
import torch
from torch import distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_probabilisticOversampling(nnUNetTrainer):
    """
    sampling of foreground happens randomly and not for the last 33% of samples in a batch
    since most trainings happen with batch size 2 and nnunet guarantees at least one fg sample, effectively this can
    be 50%
    Here we compute the actual oversampling percentage used by nnUNetTrainer in order to be as consistent as possible.
    If we switch to this oversampling then we can keep it at a constant 0.33 or whatever.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.probabilistic_oversampling = True
        self.oversample_foreground_percent = float(np.mean(
            [not sample_idx < round(self.configuration_manager.batch_size * (1 - self.oversample_foreground_percent))
             for sample_idx in range(self.configuration_manager.batch_size)]))
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.configuration_manager.batch_size
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = [global_batch_size // world_size] * world_size
            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                  if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                  else batch_size_per_GPU[i]
                                  for i in range(len(batch_size_per_GPU))]
            assert sum(batch_size_per_GPU) == global_batch_size
            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])
            print("worker", my_rank, "oversample", self.oversample_foreground_percent)

            self.batch_size = batch_size_per_GPU[my_rank]


class nnUNetTrainer_probabilisticOversampling_033(nnUNetTrainer_probabilisticOversampling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.33
    
    
class nnUNetTrainer_probabilisticOversampling_010(nnUNetTrainer_probabilisticOversampling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.1
