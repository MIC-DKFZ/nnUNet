import os
from os.path import join

import nnunetv2
from nnunetv2.paths import nnUNet_extTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def recursive_find_trainer_class_by_name(trainer_name: str):
    # Import here is necessary to avoid circular import
    # this function is used in the training and inference scripts
    # but the inference script needs to import the trainer class
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
        nnunetv2.__path__[0],
    )

    if nnunet_trainer is None:
        if nnUNet_extTrainer:
            ext_paths = nnUNet_extTrainer.split(os.pathsep)
            print(
                f"Trainer '{trainer_name}' not found in nnunetv2.training.nnUNetTrainer.\n"
                f"Searching in external trainer paths from environment variable 'nnUNet_extTrainer'..."
            )
            for path in ext_paths:
                if path.strip() and os.path.exists(path):
                    print(f"Searching in: {path}")
                    nnunet_trainer = recursive_find_python_class(
                        path, trainer_name, None, base_folder=path, verbose=True
                    )
                    if nnunet_trainer is not None:
                        print(f"Using trainer '{trainer_name}' from: {path}")
                        break
        if nnunet_trainer is None:
            raise RuntimeError(
                f"Could not find requested nnunet trainer {trainer_name} in "
                f"nnunetv2.training.nnUNetTrainer ("
                f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}).'
                f"If the trainer is located elsewhere, please move it there or specify the external path via the "
                f"`nnUNet_extTrainer` environment variable."
                f"nnUNet_extTrainer={os.environ.get('nnUNet_extTrainer', '')}"
            )
    assert issubclass(nnunet_trainer, nnUNetTrainer), (
        "The requested nnunet trainer class must inherit from 'nnUNetTrainer'"
    )
    return nnunet_trainer
