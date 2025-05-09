from typing import Union, Tuple
import argparse
import numpy as np
import os
from os.path import join
from pathlib import Path
from time import time
import torch
from torch._dynamo import OptimizedModule

from nnunetv2.utilities.label_handling.label_handling import LabelManager

from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient


class FlarePredictor(nnUNetPredictor):
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            self.network = torch.compile(self.network)


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    # predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    segmentation = predicted_logits.argmax(0)
    del predicted_logits
    # segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    torch.set_num_threads(old_threads)
    return segmentation_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    segmentation_final = ret

    rw = NibabelIOWithReorient()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def predict_flare(input_dir, output_dir, model_folder, folds=("all",)):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_files = sorted(input_dir.glob("*.nii.gz"))
    output_files = [str(output_dir / f.name[:-12]) for f in input_files]
    for input_file, output_file in zip(input_files, output_files):
        print(f"Predicting {input_file.name}")
        start = time()
        plans_manager = PlansManager(load_json(join(model_folder, 'plans.json')))
        configuration_manager = plans_manager.get_configuration("3d_fullres")
        dataset_json = load_json(join(model_folder, 'dataset.json'))
        rw = NibabelIOWithReorient()
        image, props = rw.read_images([input_file,])
        with torch.no_grad():
            predictor = FlarePredictor(tile_step_size=0.5, use_mirroring=False)
            predictor.initialize_from_trained_model_folder(model_folder, use_folds=folds)
            preprocessor = configuration_manager.preprocessor_class(verbose=False)
            data, _ = preprocessor.run_case_npy(image,
                                                None,
                                                props,
                                                plans_manager,
                                                configuration_manager,
                                                dataset_json)
            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
            predicted_logits = predictor.predict_logits_from_preprocessed_data(data).cpu()
            export_prediction_from_logits(predicted_logits, props, configuration_manager,
                                            plans_manager, dataset_json, output_file,
                                            False)
        print(f"Prediction time: {time() - start:.2f}s")


if __name__ == '__main__':
    os.environ['nnUNet_compile'] = 'f'
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="/workspace/inputs")
    parser.add_argument("-o", "--output", default="/workspace/outputs")
    parser.add_argument("-m", "--model", default="/opt/app/_trained_model")
    parser.add_argument("-f", "--folds", nargs="+", default=["all"])
    args = parser.parse_args()
    predict_flare(args.input, args.output, args.model, args.folds)