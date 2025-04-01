import multiprocessing
import os
import socket
from typing import Union, Optional
import yaml
import logging

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.checkpointing import check_object_exists
from torch.backends import cudnn

import mlflow


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
                          device: torch.device = torch.device('cuda'),
                          **kwargs):
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:
        raise RuntimeError(f'Could not find requested nnunet trainer {trainer_name} in '
                           f'nnunetv2.training.nnUNetTrainer ('
                           f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                    'nnUNetTrainer'

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
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, device=device, **kwargs)
    return nnunet_trainer


def load_checkpoint_from_s3(nnunet_trainer: nnUNetTrainer, final: bool = False):
    if not nnunet_trainer.mlflow_run_id is None:
        if not nnunet_trainer.checkpointing_bucket is None:
            object_name = f"checkpoints/run_{nnunet_trainer.mlflow_run_id}/checkpoint_final.pth"
            if check_object_exists(nnunet_trainer.checkpointing_bucket, object_name):
                nnunet_trainer.load_checkpoint(
                    join(nnunet_trainer.output_folder, 'checkpoint_final.pth'),
                    nnunet_trainer.mlflow_run_id
                )
                return True
            # If other checkpoints should be loaded than the final one, try to load them
            if not final:
                object_name = f"checkpoints/run_{nnunet_trainer.mlflow_run_id}/checkpoint_latest.pth"
                if check_object_exists(nnunet_trainer.checkpointing_bucket, object_name):
                    nnunet_trainer.load_checkpoint(
                        join(nnunet_trainer.output_folder, 'checkpoint_latest.pth'),
                        nnunet_trainer.mlflow_run_id
                    )
                    return True
                object_name = f"checkpoints/run_{nnunet_trainer.mlflow_run_id}/checkpoint_best.pth"
                if check_object_exists(nnunet_trainer.checkpointing_bucket, object_name):
                    nnunet_trainer.load_checkpoint(
                        join(nnunet_trainer.output_folder, 'checkpoint_best.pth'),
                        nnunet_trainer.mlflow_run_id
                    )
                    return True
            nnunet_trainer.print_to_log_file(f"WARNING: could not find the checkpoint associated with"
                                             f"mlflow_run_id: {nnunet_trainer.mlflow_run_id} on"
                                             f"bucket: {nnunet_trainer.checkpointing_bucket} with"
                                             f"object name: {object_name}.")
    return False


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer,
                          continue_training: bool,
                          validation_only: bool,
                          pretrained_weights_file: str = None
                          ):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('Cannot both continue a training AND load pretrained weights. Pretrained weights can only '
                           'be used at the beginning of the training.')
    if continue_training:
        # Try to load from S3
        if load_checkpoint_from_s3(nnunet_trainer):
            return

        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                               f"continue from. Starting a new training...")
            expected_checkpoint_file = None
    elif validation_only:
        # Try to load from S3
        if load_checkpoint_from_s3(nnunet_trainer, final=True):
            return
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


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
                 device: torch.device = torch.device('cuda'),
                 **kwargs):
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

    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus),
                 nprocs=num_gpus,
                 join=True)
    else:
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, device=device, **kwargs)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


# ---- Added as of improved argument passing requirement ----
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def get_type(type_str):
    if type_str == 'int':
        return int
    elif type_str == 'float':
        return float
    elif type_str == 'bool':
        return bool
    elif type_str == 'str':
        return str
    else:
        raise ValueError(f"Unsupported type: {type_str}")

# -----------------------------------------------------------


def run_training_entry(testing: bool = False):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    import argparse
    parser = argparse.ArgumentParser(description="nnUNet training script")
    parser.add_argument('--config', type=str,
                        help='Path to the configuration file')
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

    # ---- Added as of improved argument passing requirement ----
    # Parse known args first to check for config file
    args, _ = parser.parse_known_args()

    # If config file is provided, add those arguments
    config = {}  # Initialize config as an empty dictionary
    if args.config:
        config = load_config(args.config)
        for arg_name, arg_data in config.items():
            if f'--{arg_name}' not in parser._option_string_actions:
                arg_type = get_type(arg_data['type'])
                try:
                    default_value = arg_type(arg_data['value'])
                except ValueError as e:
                    raise ValueError(
                        f"Invalid value for {arg_name}: {arg_data['value']}. Expected type: {arg_data['type']}") from e

                parser.add_argument(f'--{arg_name}',
                                    type=arg_type,
                                    default=default_value,
                                    help=arg_data['description'])

    args = parser.parse_args()

    # Type checking
    logger.info("Performing type checks:")
    for arg_name, arg_data in config.items():
        if hasattr(args, arg_name):
            expected_type = get_type(arg_data['type'])
            actual_value = getattr(args, arg_name)
            if not isinstance(actual_value, expected_type):
                logger.warning(
                    f"  Type mismatch for {arg_name}: expected {expected_type.__name__}, got {type(actual_value).__name__}")
            else:
                logger.info(f"  {arg_name}: type check passed ({expected_type.__name__})")

    # Log all parameters and their values
    logger.info("Script parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # -----------------------------------------------------------

    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        if not testing:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    # ---- Added as of improved argument passing requirement ----
    # Convert args to a dictionary
    args_dict = vars(args)

    # Remove the 'config' key from args_dict
    args_dict.pop('config', None)

    # Rename some keys to match the function parameters
    args_dict['trainer_class_name'] = args_dict.pop('tr', False)
    args_dict['plans_identifier'] = args_dict.pop('p', False)
    args_dict['export_validation_probabilities'] = args_dict.pop('npz', False)
    args_dict['continue_training'] = args_dict.pop('c', False)
    args_dict['only_run_validation'] = args_dict.pop('val', False)
    args_dict['val_with_best'] = args_dict.pop('val_best', False)

    # Convert device string to torch.device
    args_dict['device'] = torch.device(device)

    # run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
    #              args.num_gpus, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
    #              device=device)

    # Call run_training with unpacked dictionary
    if not testing:
        if 'TRACKING_URI' in os.environ and 'SECRET_KEY' in os.environ:
            # Try to config MLFlow client with JWT authentication
            try:
                from mlflow_jwt_auth import configure_mlflow
                configure_mlflow()
                print(f"Successfully configured JWT aut for MLFlow")
            except Exception as e:
                print(f"Failed to configure MLflow JWT auth: {e}")
        else:
            # Try to config MLFlow client with directly tracking_uri
            try:
                mlflow.set_tracking_uri(args_dict['tracking_uri'])
                print(f"Successfully configured MLFlow tracking uri: {args_dict['tracking_uri']}")
            except Exception as e:
                print(f"Failed to configure MLFlow tracking uri: {e}")

        # Setting up experiments if it does not exist already
        try:
            experiment_name = args_dict['experiment_name']
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
            else:
                print(f"Experiment '{experiment_name}' already exists with ID: {experiment.experiment_id}")
                print(experiment)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Failed to setup MLFlow experiment: {e}")

        # Start training either with or without the mlflow tracking
        if not mlflow.get_experiment_by_name(experiment_name) is None:
            with mlflow.start_run(
                args_dict['mlflow_run_id'] if len(args_dict['mlflow_run_id']) > 0 else None
            ):
                run_training(**args_dict)
        else:
            run_training(**args_dict)

    # -----------------------------------------------------------


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    # multiprocessing.set_start_method("spawn")
    run_training_entry()
