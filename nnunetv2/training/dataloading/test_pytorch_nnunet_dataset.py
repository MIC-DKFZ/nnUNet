"""
python nnUNet/nnunetv2/training/dataloading/test_pytorch_nnunet_dataset.py \
    --dataset_id=58 \
    --configuration=3d_fullres \
    --fold=2 \
    --num_gpus_available_to_ddp=4 \
    --global_batch_size=16 \
    --prefetch_factor=2 \
    --global_oversample_foreground_percent=0.33 \
    --num_epochs=100 \
    --num_iterations_per_epoch=250 \
    --num_dataloader_workers=4
"""

import argparse
import os
import os.path as osp
from typing import Dict, List

import numpy as np
import psutil
import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities import file_and_folder_operations as nnunet_file_utils
from blib.logging import logger
from torch.utils.data import DataLoader

from nnunetv2 import paths as nnunet_paths
from nnunetv2.run import run_training as run_utils
from nnunetv2.training.data_augmentation import (
    compute_initial_patch_size as patch_utils,
)
from nnunetv2.training.dataloading.pytorch_nnunet_dataset import nnUNetPytorchDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities import dataset_name_id_conversion as nnunet_dataset_id_utils
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

log = logger.get_logger(__name__)


def get_current_cpu_id() -> str:
    process = psutil.Process(os.getpid())
    cpu_affinity = process.cpu_affinity()
    current_cpu_id = os.sched_getaffinity(0)
    return current_cpu_id


class MiniNNUNetDDPTrainer:
    """Implements the parts of nnUNetTrainer relevant to the dataloading process."""

    # Whether to use distributed data parallel
    is_ddp: bool
    # DDP worker ID
    local_rank: int
    # Device to use
    device: torch.device
    # Dataset plans
    plans_manager: PlansManager
    # Labels
    label_manager: LabelManager
    # Configuration (e.g. batch size, patch size, etc.)
    configuration_manager: ConfigurationManager

    # Train split
    train_split: List[str]
    batch_size: int
    prefetch_factor: int
    oversample_foreground_percent: float
    num_dataloader_workers: int
    # Path to the preprocessed dataset folder containing .npy, .npz, .pkl, and _seg.npy files
    preprocessed_dataset_data_folder: str

    train_dataloader: torch.utils.data.DataLoader  # initialized in `on_train_start`

    def __init__(
        self,
        plans: Dict,
        configuration: str,
        dataset: Dict,
        train_split_keys: List[str],
        batch_size: int,
        prefetch_factor: int,
        oversample_foreground_percent: float,
        num_dataloader_workers: int,
        preprocessed_dataset_folder: str,
    ) -> None:
        """Initializes `MiniNNUNetTrainer`.

        Based on `nnUNetTrainer.__init__`.

        Args:
            dataset: A dictionary containing the dataset information, resembles:
                {
                    "channel_names": {
                        "0": "CT"
                    },
                    "labels": {
                        "background": 0,
                        "CAC + Stent": [
                            1,
                            2
                        ],
                        "CAC": 1
                    },
                    "regions_class_order": [
                        1,
                        2
                    ],
                    "numTraining": 1661,
                    "file_ending": ".nii.gz"
                }
            train_split_keys: A list of datapoints. In the original implementation, `fold` is an arg
                of the trainer and the train and validation splits are constructed internal to
                the class. Here, it's simpler to pass the split directly since we do not need both
                splits.
            batch_size: The batch size for this particular DDP worker e.g. if `--global_batch_size`
                is 16 and `--num_gpus_available_to_ddp` is 2, then batch size should be 8.
            preprocessed_dataset_folder: The path to the folder containing the data for the
                preprocessed dataset, resembles:

                +- {preprocessed_dataset_folder}
                    +- nnUNetPlans.json
                    +- splits_final.json
                    +- gt_segmentations/
                    |  +- {dataset_name}_000.nii.gz
                    |  +- {dataset_name}_001.nii.gz
                    |  +- {dataset_name}_002.nii.gz
                    |  +- ...
                    +- {data_identifier}/
                        +- {dataset_name}_000.npy
                        +- {dataset_name}_000.npz
                        +- {dataset_name}_000.pkl
                        +- {dataset_name}_000_seg.npy
                        +- {dataset_name}_001.npy
                        +- {dataset_name}_001.npz
                        +- {dataset_name}_001.pkl
                        +- {dataset_name}_001_seg.npy
                        +- {dataset_name}_002.npy
                        +- {dataset_name}_002.npz
                        +- ...

                {preprocessed_dataset_folder} is the join of {nnUNet_preprocessed}/{dataset_name}
                where {nnUNet_preprocessed} originates from the $nnUNet_preprocessed environment
                variable and accessed via `nnunet_paths.nnUNet_preprocessed`, {dataset_name}
                originates from the CLI argument `--dataset_id` and converted to a dataset name via
                `nnunet_dataset_id_utils.maybe_convert_to_dataset_name`, and {data_identifier}
                originates from the CLI argument `--configuration` and converted to an identifier
                via `self.configuration_manager.data_identifier`.

                Example:
                    {nnUNet_preprocessed}: "/data/nnUNet/nnUNet_preprocessed"
                    {dataset_name}: "Dataset048_NonContrastOneCAC_1291_plus_segmedDeviceEnriched_plus_stenteEnrichedProspective_CAC_and_StentLabels"
                    {data_identifier}: "nnUNetPlans_3d_fullres"

                Note: the original implementation uses the convention

                    +- {preprocessed_dataset_folder_base}
                        +- nnUNetPlans.json
                        +- splits_final.json
                        +- gt_segmentations/
                        +- {data_identifier}/

                and {preprocessed_dataset_folder} is the join of
                {preprocessed_dataset_folder_base}/{data_identifier}.
        """
        self.is_ddp = dist.is_available() and dist.is_initialized()
        if not self.is_ddp:
            raise NotImplementedError("Only DDP is supported")
        self.local_rank = dist.get_rank()
        self.device = torch.device(type="cuda", index=self.local_rank)
        log.info(
            "Using DDP",
            local_rank=self.local_rank,
            device_count=torch.cuda.device_count(),  # number of GPUs
            world_size=dist.get_world_size(),  # number of GPUs in this process group
        )
        self.plans_manager = PlansManager(plans)
        self.label_manager = self.plans_manager.get_label_manager(dataset)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.train_split_keys = train_split_keys
        self.batch_size = int(batch_size)
        self.prefetch_factor = int(prefetch_factor)
        self.oversample_foreground_percent = float(oversample_foreground_percent)
        self.num_dataloader_workers = num_dataloader_workers
        self.preprocessed_dataset_data_folder = osp.join(
            preprocessed_dataset_folder,
            self.configuration_manager.data_identifier,
        )

    def run_training(
        self,
        num_epochs: int,
        num_iterations_per_epoch: int,
    ) -> None:
        self.on_train_start()
        for epoch in range(num_epochs):
            for batch_id in range(num_iterations_per_epoch):
                log.info(
                    "Loading batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    rank=self.local_rank,
                    cpu_id=get_current_cpu_id()
                )
                batch = next(iter(self.train_dataloader))
                log.info(
                    "Loaded batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    rank=self.local_rank,
                    cpu_id=get_current_cpu_id()
                )
                self.train_step(batch)

    def on_train_start(self) -> None:
        self.train_dataloader = self.get_train_dataloader()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> None:
        pass  # no-op

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )

    def get_train_dataset(self) -> torch.utils.data.Dataset:
        # instead of `configure_rotation_dummyDA_mirroring_and_inital_patch_size``
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA = {
            "x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "y": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            "z": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
        }
        do_dummy_2d_data_aug = False
        initial_patch_size = patch_utils.get_patch_size(
            patch_size,
            *rotation_for_DA.values(),
            (0.85, 1.25),
        )
        mirror_axes = (0, 1, 2)
        # instead of `_get_deep_supervision_scales`
        pool_op_kernel_sizes = self.configuration_manager.pool_op_kernel_sizes
        scales_arr = 1 / np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)
        deep_supervision_scales = list(list(i) for i in scales_arr)[:-1]

        train_transforms = nnUNetTrainer.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=False,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None,
            ignore_label=self.label_manager.ignore_label,
        )
        return nnUNetPytorchDataset(
            self.preprocessed_dataset_data_folder,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            train_transforms,
            self.train_split_keys,
            oversample_foreground_percent=self.oversample_foreground_percent,
            folder_with_segs_from_previous_stage=None,
            num_images_properties_loading_threshold=0,
        )


def compute_oversample_foreground_percent(
    global_oversample_foreground_percent: float,
    rank: int,
    num_gpus_available_to_ddp: int,
) -> float:
    assert rank < num_gpus_available_to_ddp
    low = rank / num_gpus_available_to_ddp
    high = (rank + 1) / num_gpus_available_to_ddp
    if high < (1 - global_oversample_foreground_percent):
        return 0.0
    if low > (1 - global_oversample_foreground_percent):
        return 1.0
    numerator = 1 - global_oversample_foreground_percent - low
    denominator = high - low
    # 1- since we oversample in the last part of the batch instead of the first part
    return 1 - numerator / denominator


def get_trainer(
    dataset_id: int,
    configuration: str,
    fold: int,
    batch_size: int,
    prefetch_factor: int,
    oversample_foreground_percent: float,
    num_dataloader_workers: int,
) -> MiniNNUNetDDPTrainer:
    ...
    preprocessed_dataset_folder = osp.join(
        nnunet_paths.nnUNet_preprocessed,
        nnunet_dataset_id_utils.maybe_convert_to_dataset_name(dataset_id),
    )
    plans = nnunet_file_utils.load_json(
        osp.join(preprocessed_dataset_folder, "nnUNetPlans.json")
    )
    splits = nnunet_file_utils.load_json(
        osp.join(preprocessed_dataset_folder, "splits_final.json")
    )
    dataset = nnunet_file_utils.load_json(
        osp.join(preprocessed_dataset_folder, "dataset.json")
    )
    return MiniNNUNetDDPTrainer(
        plans=plans,
        configuration=configuration,
        dataset=dataset,
        train_split_keys=splits[fold]["train"],
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        oversample_foreground_percent=oversample_foreground_percent,
        num_dataloader_workers=num_dataloader_workers,
        preprocessed_dataset_folder=preprocessed_dataset_folder,
    )


def run_ddp(
    rank: int,
    dataset_id: int,
    configuration: str,
    fold: int,
    num_gpus_available_to_ddp: int,
    global_batch_size: int,
    prefetch_factor: int,
    global_oversample_foreground_percent: float,
    num_dataloader_workers: int,
    num_epochs: int,  # hardcoded to 1000 in nnUNetTrainer
    num_iterations_per_epoch: int,  # hardcoded to 250 in nnUNetTrainer
) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=num_gpus_available_to_ddp)
    torch.cuda.set_device(torch.device("cuda", dist.get_rank()))
    oversample_foreground_percent = compute_oversample_foreground_percent(
        global_oversample_foreground_percent,
        rank,
        num_gpus_available_to_ddp,
    )
    trainer = get_trainer(
        dataset_id,
        configuration,
        fold,
        global_batch_size // num_gpus_available_to_ddp,
        prefetch_factor,
        oversample_foreground_percent,
        num_dataloader_workers,
    )
    trainer.run_training(num_epochs, num_iterations_per_epoch)


def main(args: argparse.Namespace) -> None:
    assert args.global_batch_size % args.num_gpus_available_to_ddp == 0
    # For DDP, set the MASTER_ADDR and MASTER_PORT environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(run_utils.find_free_network_port())
    mp.spawn(
        run_ddp,
        args=(
            args.dataset_id,
            args.configuration,
            args.fold,
            args.num_gpus_available_to_ddp,
            args.global_batch_size,
            args.prefetch_factor,
            args.global_oversample_foreground_percent,
            args.num_dataloader_workers,
            args.num_epochs,
            args.num_iterations_per_epoch,
        ),
        nprocs=args.num_gpus_available_to_ddp,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=int, required=True, default="59")
    parser.add_argument(
        "--configuration", type=str, required=True, default="3d_fullres"
    )
    parser.add_argument("--fold", type=int, required=True, default=2)
    parser.add_argument(
        "--num_gpus_available_to_ddp", type=int, required=True, default=4
    )

    # parser.add_argument("--tr", type=str, required=True, default="nnUNetTrainerDiceCELoss_noSmooth")
    # parser.add_argument("--p", type=str, required=True, default="nnUNetPlans")
    # parser.add_argument("--pretrained_weights", type=str, required=False, default=None)

    parser.add_argument("--global_batch_size", type=int, required=True, default=16)
    parser.add_argument("--prefetch_factor", type=int, required=True, default=2)
    parser.add_argument(
        "--global_oversample_foreground_percent",
        type=float,
        required=True,
        default=0.33,
    )
    parser.add_argument("--num_epochs", type=int, required=True, default=1)
    parser.add_argument(
        "--num_iterations_per_epoch", type=int, required=True, default=10
    )
    parser.add_argument("--num_dataloader_workers", type=int, required=True, default=4)

    main(parser.parse_args())
