import json
from os.path import isdir
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import onnx
import onnxruntime
import torch
from torch._dynamo import OptimizedModule

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.dataset_name_id_conversion import \
    maybe_convert_to_dataset_name
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
    output_names: tuple[str, ...] = None,
    verbose: bool = False,
) -> None:
    if not output_names:
        output_names = (f"{checkpoint[:-4]}.onnx" for checkpoint in save_checkpoints)

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    for c in configurations:
        print(f"Configuration {c}")
        trainer_output_dir = get_output_folder(
            dataset_name, trainer, plans_identifier, c
        )

        if not isdir(trainer_output_dir):
            if strict:
                raise RuntimeError(
                    f"{dataset_name} is missing the trained model of configuration {c}"
                )
            else:
                print(f"Skipping configuration {c}, does not exist")
                continue

        predictor = nnUNetPredictor(
            perform_everything_on_gpu=False,
            device=torch.device("cpu"),
        )

        for checkpoint_name, output_name in zip(save_checkpoints, output_names):
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

                network.eval()

                curr_output_dir = output_dir / c / f"fold_{fold}"
                if not curr_output_dir.exists():
                    curr_output_dir.mkdir(parents=True)
                else:
                    if len(list(curr_output_dir.iterdir())) > 0:
                        raise RuntimeError(
                            f"Output directory {curr_output_dir} is not empty"
                        )

                rand_input = torch.rand((1, 1, *config.patch_size))
                torch_output = network(rand_input)

                torch.onnx.export(
                    network,
                    rand_input,
                    curr_output_dir / output_name,
                    export_params=True,
                    verbose=verbose,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )

                onnx_model = onnx.load(curr_output_dir / output_name)
                onnx.checker.check_model(onnx_model)

                ort_session = onnxruntime.InferenceSession(
                    curr_output_dir / output_name, providers=["CPUExecutionProvider"]
                )
                ort_inputs = {ort_session.get_inputs()[0].name: rand_input.numpy()}
                ort_outs = ort_session.run(None, ort_inputs)

                np.testing.assert_allclose(
                    torch_output.detach().cpu().numpy(),
                    ort_outs[0],
                    rtol=1e-03,
                    atol=1e-05,
                )

                print(
                    f"Successfully exported and verified {curr_output_dir / output_name}"
                )

                with open(curr_output_dir / "config.json", "w") as f:
                    json.dump(
                        {
                            "dataset_name": dataset_name,
                            "configuration": c,
                            "trainer": trainer,
                            "plans_identifier": plans_identifier,
                            "fold": fold,
                            "checkpoint_name": checkpoint_name,
                            "configuration_manager": {
                                k: config.configuration[k]
                                for k in [
                                    "patch_size",
                                    "spacing",
                                    "normalization_schemes",
                                    # These are mostly interesting for certification
                                    # uses, but they are also useful for debugging.
                                    "UNet_class_name",
                                    "UNet_base_num_features",
                                    "unet_max_num_features",
                                    "conv_kernel_sizes",
                                    "pool_op_kernel_sizes",
                                    "num_pool_per_axis",
                                ]
                            },
                        },
                        f,
                        indent=4,
                    )
