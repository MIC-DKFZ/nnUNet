import torch

from nnunetv2.training.nnUNetTrainer.variants.benchmarking.nnUNetTrainerBenchmark_5epochs import (
    nnUNetTrainerBenchmark_5epochs,
)
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainerBenchmark_5epochs_noDataLoading(nnUNetTrainerBenchmark_5epochs):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._set_batch_size_and_oversample()
        num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        patch_size = self.configuration_manager.patch_size
        dummy_data = torch.rand((self.batch_size, num_input_channels, *patch_size), device=self.device)
        if self.enable_deep_supervision:
            dummy_target = [
                torch.round(
                    torch.rand((self.batch_size, 1, *[int(i * j) for i, j in zip(patch_size, k)]), device=self.device)
                    * max(self.label_manager.all_labels)
                )
                for k in self._get_deep_supervision_scales()
            ]
        else:
            raise NotImplementedError("This trainer does not support deep supervision")
        self.dummy_batch = {"data": dummy_data, "target": dummy_target}

    def get_dataloaders(self):
        return None, None

    def run_training(self):
        try:
            self.on_train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []
                for batch_id in range(self.num_iterations_per_epoch):
                    train_outputs.append(self.train_step(self.dummy_batch))
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(self.dummy_batch))
                    self.on_validation_epoch_end(val_outputs)

                self.on_epoch_end()

            self.on_train_end()
        except RuntimeError:
            self.crashed_with_runtime_error = True
