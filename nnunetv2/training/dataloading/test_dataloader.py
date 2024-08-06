import os
from typing import List

import numpy as np
import shutil

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.pytorch_nnunet_dataset import nnUNetPytorchDataset

import os
import socket
from typing import Union, Optional

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
from torch.backends import cudnn

from nnunetv2.run.run_training import get_trainer_from_args, find_free_network_port


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def run_ddp_dataloading(rank,  dataset_name_or_id, configuration, fold, tr, p, use_compressed, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    # Based on the trainer we use a different dataloader will be used
    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p,
                                        use_compressed)
    
    print("Trainer used is: ", nnunet_trainer)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    nnunet_trainer.run_dataloading_test()



def run_dataloader_test(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 use_compressed_data: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda')):
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

        mp.spawn(run_ddp_dataloading,
                 args=(
                     dataset_name_or_id,                     
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     use_compressed_data,
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
                                            plans_identifier, use_compressed_data, device=device)

        
        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        nnunet_trainer.run_dataloading_test()


def run_dataloader_test_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer. To run the Pytorch DataLoader, use "nnUNetTrainerPyTorchDataloader"')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument('-device', type=str, default='cuda', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_dataloader_test(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p,
                 num_gpus=args.num_gpus, use_compressed_data=args.use_compressed, device=device)


if __name__ == '__main__':
    run_dataloader_test_entry()
