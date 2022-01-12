import argparse
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import join

import nnunetv2
from nnunetv2.paths import default_plans_identifier
from nnunetv2.training.nnunet_modules.nnUNetModule import nnUNetModule
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import pytorch_lightning as pl


def nnUNet_train(trainer_class_name: str, dataset_name_or_id: Union[int, str], plans_name: str, configuration: str,
                 fold: int, unpack_dataset: bool = True, folder_with_segs_from_previous_stage: str = None,
                 pretrained_weights_from_checkpoint: str = None, resume: bool = False, num_gpus: int = 1):
    trainer_class_name = 'nnUNetModule'
    dataset_name_or_id = 2
    plans_name = default_plans_identifier
    configuration = '3d_fullres'
    fold = 0
    unpack_dataset = True
    folder_with_segs_from_previous_stage = None
    pretrained_weights_from_checkpoint = None
    resume = None
    num_gpus = 1

    nnunet_module = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnunet_modules"), trainer_class_name, 'nnunetv2.training.nnunet_modules')
    if nnunet_module is None:
        raise RuntimeError(f'Could not find requested nnunet module {trainer_class_name} in '
                           f'nnunetv2.training.nnunet_modules ('
                           f'{join(nnunetv2.__path__[0], "training", "nnunet_modules")})')
    assert issubclass(nnunet_module, nnUNetModule), 'The requested nnunet module class must inherit from ' \
                                                    'nnUNetModule. Please also note that the __init__ must accept ' \
                                                    'the same input aruguments (in the correct order)'

    nnunet_module = nnunet_module(dataset_name_or_id, plans_name, configuration, fold, unpack_dataset,
                                  folder_with_segs_from_previous_stage)

    # Todo pretrained weights and resuming from checkpoint
    trainer = pl.Trainer(logger=None, default_root_dir=nnunet_module.output_folder,
                         gradient_clip_val=None,
                         gradient_clip_algorithm='norm', num_nodes=1, gpus=num_gpus, enable_progress_bar=False,
                         sync_batchnorm=num_gpus > 1, precision=16,
                         resume_from_checkpoint=None,  # TODO
                         benchmark=True, deterministic=False, #  nnunet dataloaders are nondeterminstic regardless of what you set here!
                         replace_sampler_ddp=False,
                         max_epochs=nnunet_module.num_epochs, num_sanity_val_steps=0
                         )

    tr_gen, val_gen = nnunet_module.get_dataloaders()
    trainer.fit(nnunet_module, tr_gen, val_gen)


def nnUNet_train_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('')