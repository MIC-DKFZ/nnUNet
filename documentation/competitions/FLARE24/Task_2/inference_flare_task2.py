from typing import Union, List, Tuple
import argparse
import itertools
import multiprocessing
import numpy as np
import os
from os.path import join
from pathlib import Path
from time import time
import torch
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import openvino as ov
import openvino.properties.hint as hints

from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.utilities.label_handling.label_handling import LabelManager
import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


torch.set_num_threads(multiprocessing.cpu_count())


class FlarePredictor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = False,
                 device: torch.device = torch.device('cpu'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 ):
        super().__init__(tile_step_size, use_gaussian, use_mirroring, perform_everything_on_device, device, verbose,
                         verbose_preprocessing, allow_tqdm)
        if self.device == torch.device('cuda') or self.device == 'cuda':
            raise RuntimeError('CUDA is not supported for this task')

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth',
                                             save_model: bool = True):
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
        assert len(use_folds) == 1, 'Only one fold is supported for this task'

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
        
            if save_model:
                parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        if save_model:
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
            self.network = network
            self.allowed_mirroring_axes = inference_allowed_mirroring_axes

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if save_model:
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(parameters[0])
            else:
                self.network._orig_mod.load_state_dict(parameters[0])
            self.network.eval()

        config = {hints.performance_mode: hints.PerformanceMode.LATENCY,
                 hints.enable_cpu_pinning(): True,
                 }
        core = ov.Core()
        core.set_property(
            "CPU",
            {hints.execution_mode: hints.ExecutionMode.PERFORMANCE},
        )
        if save_model:
            input_tensor = torch.randn(1, num_input_channels, *configuration_manager.patch_size, requires_grad=False)
            ov_model = ov.convert_model(self.network, example_input=input_tensor)
            ov.save_model(ov_model, f"{model_training_output_dir}/model.xml")
            import sys
            sys.exit(0)
        ov_model = core.read_model(f"{model_training_output_dir}/model.xml")
        self.network = core.compile_model(ov_model, "CPU", config= config)

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = 1,
                           num_processes_segmentation_export: int = 1,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """

        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)
        
        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        if self.use_openvino:
            prediction = torch.from_numpy(self.network(x)[0])
        else:
            prediction = self.network(x)

        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                if not self.is_openvino:
                    prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
                else:
                    temp_pred = torch.from_numpy(self.network(torch.flip(x, axes))[0])
                    prediction += torch.flip(temp_pred, axes)

            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                        dtype=torch.half,
                                        device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        prediction = None

        if not self.use_openvino:
            for params in self.list_of_parameters:

                # messing with state dict names...
                if not isinstance(self.network, OptimizedModule):
                    self.network.load_state_dict(params)
                else:
                    self.network._orig_mod.load_state_dict(params)

                # why not leave prediction on device if perform_everything_on_device? Because this may cause the
                # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
                # this actually saves computation time
                if prediction is None:
                    prediction = self.predict_sliding_window_return_logits(data).to('cpu')
                else:
                    prediction += self.predict_sliding_window_return_logits(data).to('cpu')

            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)

        else:
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data)
            else:
                prediction += self.predict_sliding_window_return_logits(data)

        if self.verbose: print('Prediction done')
        return prediction

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        if self.device not in [torch.device('cpu'), 'cpu']:
            self.network = self.network.to(self.device)
            self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose: 
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    def convert_predicted_logits_to_segmentation_with_correct_shape(self, predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                    plans_manager: PlansManager,
                                                                    configuration_manager: ConfigurationManager,
                                                                    label_manager: LabelManager,
                                                                    properties_dict: dict,
                                                                    return_probabilities: bool = False,
                                                                    num_threads_torch: int = default_num_processes):

        # resample to original shape
        current_spacing = configuration_manager.spacing if \
            len(configuration_manager.spacing) == \
            len(properties_dict['shape_after_cropping_and_before_resampling']) else \
            [properties_dict['spacing'][0], *configuration_manager.spacing]
        predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                                properties_dict['shape_after_cropping_and_before_resampling'],
                                                current_spacing,
                                                properties_dict['spacing'])
        segmentation = predicted_logits.argmax(0)
        del predicted_logits

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
        return segmentation_reverted_cropping

    def export_prediction_from_logits(self, predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                      configuration_manager: ConfigurationManager,
                                      plans_manager: PlansManager,
                                      dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                      save_probabilities: bool = False):

        if isinstance(dataset_json_dict_or_file, str):
            dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

        label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
        ret = self.convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
            return_probabilities=save_probabilities
        )
        del predicted_array_or_file

        segmentation_final = ret

        rw = plans_manager.image_reader_writer_class()
        rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                     properties_dict)

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        self.export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                        self.plans_manager, self.dataset_json, output_file_truncated,
                                        save_or_return_probabilities)


def predict_flare(input_dir, output_dir, model_folder, save_model):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    input_files = list(input_dir.glob("*.nii.gz"))
    output_files = [str(output_dir / f.name[:-12]) for f in input_files]
    for input_file, output_file in zip(input_files, output_files):
        print(f"Predicting {input_file.name}")
        start = time()
        predictor = FlarePredictor(tile_step_size=0.5, use_mirroring=False, device=torch.device("cpu"))
        predictor.initialize_from_trained_model_folder(model_folder, ("all",), save_model=save_model)
        rw = predictor.plans_manager.image_reader_writer_class()
        image, props = rw.read_images([input_file,])
        _ = predictor.predict_single_npy_array(image, props, None, output_file, False)
        print(f"Prediction time: {time() - start:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="/workspace/inputs")
    parser.add_argument("-o", "--output", default="/workspace/outputs")
    parser.add_argument("-m", "--model", default="/opt/app/_trained_model")
    parser.add_argument("-save_model", action="store_true")
    args = parser.parse_args()
    predict_flare(args.input, args.output, args.model, args.save_model)