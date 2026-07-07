import argparse
import signal
from typing import Union
import torch
import os
import multiprocessing as mp
import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.run.run_training import find_free_network_port, maybe_load_checkpoint, run_ddp, setup_ddp, cleanup_ddp
from torch.backends import cudnn
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer

def get_trainer_from_args(
        dataset_name_or_id: Union[int, str],
        configuration: str,
        fold: int,
        trainer_name: str = "nnUNetTrainer",
        plans_identifier: str = "nnUNetPlans",
        device: torch.device = torch.device("cuda"),
        pretrained_from_scratch: bool = False,
        overwrite_ckpt_path: str = None
):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer_name, "nnunetv2.training.nnUNetTrainer"
    )
    if nnunet_trainer is None:
        raise RuntimeError(
            f"Could not find requested nnunet trainer {trainer_name} in "
            f"nnunetv2.training.nnUNetTrainer ("
            f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
            f"else, please move it there."
        )
    assert issubclass(nnunet_trainer, nnUNetTrainer), (
        "The requested nnunet trainer class must inherit from " "nnUNetTrainer"
    )

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith("Dataset"):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(
                f"dataset_name_or_id must either be an integer or a valid dataset name with the pattern "
                f"DatasetXXX_YYY where XXX are the three(!) task ID digits. Your "
                f"input: {dataset_name_or_id}"
            )

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, "dataset.json"))
    if pretrained_from_scratch:
        plans["plans_name"] = plans["plans_name"] + "__from_scratch"
        plans["pretrain_info"]["checkpoint_path"] = None
    if overwrite_ckpt_path is not None:
        plans["pretrain_info"]["checkpoint_path"] = overwrite_ckpt_path
    nnunet_trainer = nnunet_trainer(
        plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, use_pretrained_weights=not pretrained_from_scratch, device=device
    )
    return nnunet_trainer

def train_pretrained(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str,
    plans_identifier: str,
    from_scratch: bool,
    num_gpus: int = 1,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: torch.device = torch.device("cuda"),
    overwrite_ckpt_path: str =None
):
    if isinstance(fold, str):
        if fold != "all":
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!'
                )
                raise e

    if val_with_best:
        assert not disable_checkpointing, "--val_best is not compatible with --disable_checkpointing"

    if num_gpus > 1:
        assert (
            device.type == "cuda"
        ), f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ["MASTER_PORT"] = port  # str(port)

        mp.spawn(
            run_ddp,
            args=(
                dataset_name_or_id,
                configuration,
                fold,
                trainer_class_name,
                plans_identifier,
                disable_checkpointing,
                continue_training,
                only_run_validation,
                None,
                export_validation_probabilities,
                val_with_best,
                num_gpus,
                from_scratch,
                overwrite_ckpt_path,
            ),
            nprocs=num_gpus,
            join=True,
        )
    else:
        # ToDo
        nnunet_trainer: PretrainedTrainer = get_trainer_from_args(
            dataset_name_or_id,
            configuration,
            fold,
            trainer_class_name,
            plans_identifier,
            device=device,
            pretrained_from_scratch=from_scratch,  # <-- Creates new plan name if true. Allows easy comparison Pretrained vs Non-Pretrained
            overwrite_ckpt_path=overwrite_ckpt_path)

        nnunet_trainer.use_pretrained_weights = False if (continue_training or from_scratch) else True

        # Prepare the auto-exiting in case wall-time is exceeded.
        #  This sets a internal flag, letting the trainer know it's 10 minutes till wall-clock time is up.
        #  Only wired up if the trainer implements exit_training (not present in the standard nnU-Net base trainer).
        if hasattr(nnunet_trainer, "exit_training"):
            signal.signal(signal.SIGUSR1, nnunet_trainer.exit_training)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (
            continue_training and only_run_validation
        ), f"Cannot set --c and --val flag at the same time. Dummy."

        # Still needed to allow continuation of incomplete (i.e. interrupted) trainings.
        # ToDo: Find a way to not pre-load the same checkpoints that get overriden by the --continue flag.
        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, None)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)

def run_ddp(
        rank,
        dataset_name_or_id,
        configuration,
        fold,
        tr,
        p,
        disable_checkpointing,
        c,
        val,
        pretrained_weights,
        npz,
        val_with_best,
        world_size,
        pretrained_from_scratch=False,
        overwrite_ckpt_path=None,
):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device("cuda", dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(
        dataset_name_or_id, configuration, fold, tr, p, pretrained_from_scratch=pretrained_from_scratch, overwrite_ckpt_path=overwrite_ckpt_path
    )

    # Prepare the auto-exiting in case wall-time is exceeded.
    #  This sets a internal flag, letting the trainer know it's 10 minutes till wall-clock time is up.
    #  Only wired up if the trainer implements exit_training (not present in the standard nnU-Net base trainer).
    if hasattr(nnunet_trainer, "exit_training"):
        signal.signal(signal.SIGUSR1, nnunet_trainer.exit_training)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f"Cannot set --c and --val flag at the same time. Dummy."

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()

def train_pretrained_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name_or_id", type=str, help="Dataset name or ID to train with")
    parser.add_argument("configuration", type=str, help="Configuration that should be trained")
    parser.add_argument(
        "fold", type=str, help="Fold of the 5-fold cross-validation.", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "all"]
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="PretrainedTrainer",
        help="[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=True,
        help="[REQUIRED] Use this to specify a custom plans identifier.",
    )
    parser.add_argument(
        "--from_scratch",
        required=False,
        action="store_true",
        help="[OPTIONAL] flag to train from scratch with same config as when loading checkpoints.",
    )
    parser.add_argument(
        "-num_gpus", type=int, default=1, required=False, help="Specify the number of GPUs to use for training"
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        required=False,
        help="[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted "
        "segmentations). Needed for finding the best ensemble.",
    )
    parser.add_argument(
        "--c", action="store_true", required=False, help="[OPTIONAL] Continue training from latest checkpoint"
    )
    parser.add_argument(
        "--val",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.",
    )
    parser.add_argument(
        "--val_best",
        action="store_true",
        required=False,
        help="[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead "
        "of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! "
        "WARNING: This will use the same 'validation' folder as the regular validation "
        "with no way of distinguishing the two!",
    )
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and "
        "you dont want to flood your hard drive with checkpoints.",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the training should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!",
    )
    parser.add_argument(
        "-overwrite_ckpt_path",
        type=str,
        default=None,
        required=False,
        help="Use this to overwrite the checkpoint",
    )
    args = parser.parse_args()

    assert args.device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
    if args.device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    train_pretrained(
        args.dataset_name_or_id,
        args.configuration,
        args.fold,
        args.tr,
        args.p,
        args.from_scratch,
        args.num_gpus,
        args.npz,
        args.c,
        args.val,
        args.disable_checkpointing,
        args.val_best,
        device=device,
        overwrite_ckpt_path=args.overwrite_ckpt_path
    )


if __name__ == "__main__":
    train_pretrained_entrypoint()
