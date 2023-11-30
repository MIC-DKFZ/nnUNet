import json
from os.path import isdir
from pathlib import Path
from typing import Tuple, Union

import torch
from torch._dynamo import OptimizedModule

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder


def export_onnx_model(
    dataset_name_or_id: Union[int, str],
    output_dir: Path,
    configurations: Tuple[str] = (
        "2d",
        "3d_lowres",
        "3d_fullres",
        "3d_cascade_fullres",
    ),
    trainer: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    folds: Tuple[Union[int, str], ...] = (0, 1, 2, 3, 4),
    strict: bool = True,
    save_checkpoints: Tuple[str, ...] = ("checkpoint_final.pth",),
) -> None:
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    for c in configurations:
        print(f"Configuration {c}")
        trainer_output_dir = get_output_folder(dataset_name, trainer, plans_identifier, c)

        if not isdir(trainer_output_dir):
            if strict:
                raise RuntimeError(f"{dataset_name} is missing the trained model of configuration {c}")
            else:
                print(f"Skipping configuration {c}, does not exist")
                continue

        predictor = nnUNetPredictor(
            perform_everything_on_gpu=False,
            device=torch.device("cpu"),
        )

        for checkpoint_name in save_checkpoints:
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=trainer_output_dir,
                use_folds=folds,
                checkpoint_name=checkpoint_name,
            )

            list_of_parameters = predictor.list_of_parameters
            network = predictor.network
            config = predictor.configuration_manager

            for fold, params in zip(folds, list_of_parameters):
                if not isinstance(network, OptimizedModule):
                    network.load_state_dict(params)
                else:
                    network._orig_mod.load_state_dict(params)

                export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
                rand_input = torch.rand((1, 1, *config.patch_size))
                traced_model = torch.onnx.dynamo_export(
                    network,
                    rand_input,
                    export_options=export_options,
                )

                curr_output_dir = output_dir / c / f"fold_{fold}"
                if not curr_output_dir.exists():
                    curr_output_dir.mkdir(parents=True)
                else:
                    if len(list(curr_output_dir.iterdir())) > 0:
                        raise RuntimeError(f"Output directory {curr_output_dir} is not empty")

                traced_model.save(str(curr_output_dir / "model.onnx"))
