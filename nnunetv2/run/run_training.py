import inspect
import multiprocessing
import os
import socket
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Union

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from torch.backends import cudnn
import torch


def launched_by_external_launcher() -> bool:
    """
    Did something outside nnU-Net already lay out the distributed environment for us? That is the case for
    `torchrun` (our supported multi-node path) and for anything else that speaks the same env var protocol,
    e.g. srun or deepspeed. In that case the launcher owns the process/GPU assignment and `-num_gpus` must not
    be used; nnUNetv2_train just joins the process group that was described to it.

    TORCHELASTIC_RUN_ID is set by torchrun and by nothing else. We additionally accept a complete RANK +
    WORLD_SIZE + LOCAL_RANK triple, but we insist on all three: a stale RANK left over in someone's shell must
    not silently turn `-num_gpus 4` into a single process job.
    """
    if 'TORCHELASTIC_RUN_ID' in os.environ:
        return True
    return all(k in os.environ for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))


_INIT_PROCESS_GROUP_SUPPORTS_DEVICE_ID = 'device_id' in inspect.signature(dist.init_process_group).parameters


def init_ddp_process_group(device: torch.device, **kwargs) -> None:
    """
    init_process_group, telling it which device this rank owns where the installed torch supports it.

    `device_id` makes NCCL form the communicator immediately instead of on the first collective, so a broken
    interconnect fails at startup rather than as a hang minutes later, and it gives c10d an explicit rank ->
    device mapping instead of one it has to infer (it also lets sub-groups use ncclCommSplit). It was added in
    torch 2.3 and we support torch >= 2.1.2, hence the check.
    """
    if _INIT_PROCESS_GROUP_SUPPORTS_DEVICE_ID:
        kwargs['device_id'] = device
    dist.init_process_group(**kwargs)


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          continue_training: bool = False,
                          device: torch.device = torch.device('cuda')):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_trainer_class_by_name(trainer_name)

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    plans["continue_training"] = continue_training
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, device=device)
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print("WARNING: Cannot continue training because there seems to be no checkpoint available to "
                  "continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError("Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


@dataclass
class TrainingRunOptions:
    """Everything that steers a training run but is not the trainer itself."""
    pretrained_weights: Optional[str] = None
    export_validation_probabilities: bool = False
    continue_training: bool = False
    only_run_validation: bool = False
    disable_checkpointing: bool = False
    val_with_best: bool = False


def execute_training(trainer_factory: Callable[[torch.device], nnUNetTrainer], device: torch.device,
                     options: TrainingRunOptions) -> None:
    """
    Build the trainer and run it. Knows nothing about how the process was started - by the time we get here the
    process group, if there is one, already exists and `device` is ours.
    """
    nnunet_trainer = trainer_factory(device)

    if options.disable_checkpointing:
        nnunet_trainer.disable_checkpointing = options.disable_checkpointing

    assert not (options.continue_training and options.only_run_validation), \
        'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, options.continue_training, options.only_run_validation,
                          options.pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not options.only_run_validation:
        nnunet_trainer.run_training()

    if options.val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(options.export_validation_probabilities)


def run_intranode_ddp(rank: int, trainer_factory: Callable[[torch.device], nnUNetTrainer], world_size: int,
                      options: TrainingRunOptions) -> None:
    """
    One worker of a `-num_gpus X` launch. mp.spawn pickles the arguments, so trainer_factory must be picklable:
    a functools.partial over a module level function is, a lambda or closure is not.
    """
    # Emulate torchrun environment variables for non-torchrun launches so that everything downstream only ever
    # has to read these four and never has to care how the job was started. Single node, so local == global.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    init_ddp_process_group(device, backend="nccl", rank=rank, world_size=world_size)

    try:
        execute_training(trainer_factory, device, options)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch_training(trainer_factory: Callable[[torch.device], nnUNetTrainer], device: torch.device,
                    num_gpus: int, options: TrainingRunOptions) -> None:
    """
    The single entry point for all three ways a training can be started, shared by nnUNetv2_train and
    nnUNetv2_train_pretrained:

    - an external launcher (torchrun and anything else speaking its env var protocol) already created the
      processes; we just join the process group it described. Single or multi node.
    - `-num_gpus X`: we spawn the workers ourselves. Single node only.
    - everything else: one plain process.
    """
    if options.val_with_best:
        assert not options.disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    external_launcher = launched_by_external_launcher()

    if external_launcher or num_gpus == 1:
        if external_launcher:
            assert device.type == 'cuda', f"DDP training is only implemented for cuda devices. Your device: {device}"
            assert num_gpus == 1, ("Your distributed environment was set up by an external launcher (torchrun or "
                                   "similar), so do not also pass -num_gpus: the launcher, not nnU-Net, decides how "
                                   "many processes there are and which GPU each one gets. Use -num_gpus only when "
                                   "starting the training directly.")
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device('cuda', local_rank)
            torch.cuda.set_device(device)
            init_ddp_process_group(device, backend='nccl', init_method='env://')
            print(f"Distributed launcher detected. [rank {os.environ.get('RANK', '?')} of "
                  f"{os.environ.get('WORLD_SIZE', '?')}] using cuda:{local_rank}")

        try:
            execute_training(trainer_factory, device, options)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    else:
        assert device.type == 'cuda', \
            f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_intranode_ddp, args=(trainer_factory, num_gpus, options), nprocs=num_gpus, join=True)


def trainer_from_args(dataset_name_or_id, configuration, fold, trainer_name, plans_identifier,
                      continue_training, device: torch.device) -> nnUNetTrainer:
    """get_trainer_from_args with device last, so that functools.partial can bind the rest."""
    return get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_name, plans_identifier,
                                 continue_training, device=device)


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda')):
    if plans_identifier == 'nnUNetPlans':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
                raise e

    trainer_factory = partial(trainer_from_args, dataset_name_or_id, configuration, fold, trainer_class_name,
                              plans_identifier, continue_training)
    options = TrainingRunOptions(pretrained_weights=pretrained_weights,
                                 export_validation_probabilities=export_validation_probabilities,
                                 continue_training=continue_training,
                                 only_run_validation=only_run_validation,
                                 disable_checkpointing=disable_checkpointing,
                                 val_with_best=val_with_best)
    launch_training(trainer_factory, device, num_gpus, options)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the training should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
                 device=device)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    # multiprocessing.set_start_method("spawn")
    run_training_entry()
