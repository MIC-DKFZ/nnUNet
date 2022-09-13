import torch
from batchgenerators.utilities.file_and_folder_operations import save_json, join, isfile, load_json

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBenchmark_5epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.disable_checkpointing = True
        self.num_epochs = 5
        assert torch.cuda.is_available(), "This only works on GPU"

    def perform_actual_validation(self, save_probabilities: bool = False):
        pass

    def save_checkpoint(self, filename: str) -> None:
        # do not trust people to remember that self.disable_checkpointing must be True for this trainer
        pass

    def on_train_end(self):
        torch_version = torch.__version__
        cudnn_version = torch.backends.cudnn.version()
        gpu_name = torch.cuda.get_device_name()
        epoch_times = [i - j for i, j in zip(self.logger.my_fantastic_logging['epoch_end_timestamps'], self.logger.my_fantastic_logging['epoch_start_timestamps'])]
        fastest_epoch = min(epoch_times)
        benchmark_result_file = join(self.output_folder, 'benchmark_result.json')
        if isfile(benchmark_result_file):
            old_results = load_json(benchmark_result_file)
        else:
            old_results = {}

        my_key = f"{cudnn_version}__{torch_version.replace(' ', '')}__{gpu_name.replace(' ', '')}"
        old_results[my_key] = {
            'torch_version': torch_version,
            'cudnn_version': cudnn_version,
            'gpu_name': gpu_name,
            'fastest_epoch': fastest_epoch,
        }
        save_json(old_results,
            join(self.output_folder, 'benchmark_result.json'))
