import nnunetv2
import pytorch_lightning as pl
from batchgenerators.utilities.file_and_folder_operations import join, isfile

from nnunetv2.training.callbacks.nnUNetCheckpoint import nnUNetCheckpoint
from nnunetv2.training.callbacks.nnUNetPlottingCallbacks import nnUNetProgressPngCallback
from nnunetv2.training.nnunet_modules.nnUNetModule import nnUNetModule
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def nnUNet_train_from_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=int,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetModule',
                        help='[OPTIONAL] Use this flag to specify a custom trainer module. Default: nnUNetModule')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
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
    args = parser.parse_args()

    # load nnunet class and do sanity checks
    nnunet_module = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnunet_modules"),
                                                args.tr, 'nnunetv2.training.nnunet_modules')
    if nnunet_module is None:
        raise RuntimeError(f'Could not find requested nnunet module {args.tr} in '
                           f'nnunetv2.training.nnunet_modules ('
                           f'{join(nnunetv2.__path__[0], "training", "nnunet_modules")}). If it is located somewhere '
                           f'else, please move it there.')
    assert issubclass(nnunet_module, nnUNetModule), 'The requested nnunet module class must inherit from ' \
                                                    'nnUNetModule. Please also note that the __init__ must accept ' \
                                                    'the same input aruguments as nnUNetModule'

    # handle dataset input. If it's an ID we need to convert to int from string
    if args.dataset_name_or_id.startswith('Dataset'):
        dataset_name_or_id = args.dataset_name_or_id
    else:
        try:
            dataset_name_or_id = int(args.dataset_name_or_id)
        except ValueError:
            raise ValueError(f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {args.dataset_name_or_id}')

    # initialize nnunet module
    nnunet_module = nnunet_module(dataset_name_or_id=dataset_name_or_id, plans_name=args.p, configuration=args.configuration,
                                  fold=args.fold, unpack_dataset=not args.use_compressed)

    if args.c:
        expected_checkpoint_file = join(nnunet_module.output_folder, 'checkpoint_latest.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_module.output_folder, 'checkpoint_best.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_module.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot continue training because there seems to be no checkpoint available to "
                               f"continue from. Please run without the --c flag.")
    else:
        expected_checkpoint_file = None

    # Todo pretrained weights and resuming from checkpoint, pretrained weights, validation, npz export, next stage predictions
    trainer = pl.Trainer(logger=False, default_root_dir=nnunet_module.output_folder,
                         enable_checkpointing=False,
                         callbacks=[nnUNetCheckpoint(), nnUNetProgressPngCallback()],
                         gradient_clip_val=12,
                         gradient_clip_algorithm='norm', num_nodes=1, gpus=1, enable_progress_bar=False,
                         sync_batchnorm=True, precision=16,
                         resume_from_checkpoint=expected_checkpoint_file,
                         benchmark=True,
                         deterministic=False,  # nnunet dataloaders are nondeterminstic regardless of what you set here!
                         replace_sampler_ddp=False,  # we use our own sampling
                         max_epochs=nnunet_module.num_epochs,
                         num_sanity_val_steps=0,
                         )

    tr_gen, val_gen = nnunet_module.get_dataloaders()
    trainer.fit(nnunet_module, tr_gen, val_gen)


if __name__ == '__main__':
    nnUNet_train_from_args()