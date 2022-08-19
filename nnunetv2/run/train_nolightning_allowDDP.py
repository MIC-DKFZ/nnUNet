from typing import Union, List, Tuple

import nnunetv2
import torch.cuda
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.train_nolightning import get_trainer_from_args, maybe_load_checkpoint
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn
import os
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_ddp(rank, args, world_size):
    setup(rank, world_size, args.master_port)

    nnunet_trainer = get_trainer_from_args(args.d, args.configuration, args.f, args.tr, args.p, args.use_compressed)

    if args.disable_checkpointing:
        nnunet_trainer.disable_checkpointing = args.disable_checkpointing

    assert not (args.c and args.val), f'Cannot set --c and --val flag at the same time. Dummy.'

    maybe_load_checkpoint(nnunet_trainer, args.c, args.val)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not args.val:
        nnunet_trainer.run_training()

    nnunet_trainer.perform_actual_validation(args.npz)
    cleanup()


def nnUNet_train_from_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=int,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training. If > 1, setting -master_port is '
                             'mandatory!')
    parser.add_argument('-master_port', required=False, type=int, default=None,
                        help='Master port to use for DDP communication. Can be any free port on your system')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    args = parser.parse_args()

    if args.num_gpus > 1:
        assert args.master_port is not None, 'If you request more than one GPU, make sure to set -master_port!'

    mp.spawn(run_ddp,
             args=(args, args.num_gpus),
             nprocs=args.num_gpus,
             join=True)


if __name__ == '__main__':
    nnUNet_train_from_args()
