import argparse
import gc
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Union, Tuple

import nnunetv2
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch._dynamo import OptimizedModule
from torch.backends import cudnn
from tqdm import tqdm


class CustomPredictor(nnUNetPredictor):
    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None):
        torch.set_num_threads(7)
        with torch.no_grad():
            self.network = self.network.to(self.device)
            self.network.eval()

            if self.verbose:
                print('preprocessing')
            preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)
            data, _ = preprocessor.run_case_npy(input_image, None, image_properties,
                                                self.plans_manager,
                                                self.configuration_manager,
                                                self.dataset_json)

            data = torch.from_numpy(data)
            del input_image
            if self.verbose:
                print('predicting')

            predicted_logits = self.predict_preprocessed_image(data)

            if self.verbose: print('Prediction done')

            segmentation = self.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits,
                                                                                            image_properties)
        return segmentation

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

            parameters.append(join(model_training_output_dir, f'fold_{f}', checkpoint_name))

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
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @torch.inference_mode(mode=True)
    def predict_preprocessed_image(self, image):
        empty_cache(self.device)
        data_device = torch.device('cpu')
        predicted_logits_device = torch.device('cpu')
        gaussian_device = torch.device('cpu')
        compute_device = torch.device('cuda:0')

        data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                   'constant', {'value': 0}, True,
                                                   None)
        del image

        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        empty_cache(self.device)

        data = data.to(data_device)
        predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                       dtype=torch.half,
                                       device=predicted_logits_device)
        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                    value_scaling_factor=10,
                                    device=gaussian_device, dtype=torch.float16)

        if not self.allow_tqdm and self.verbose:
            print(f'running prediction: {len(slicers)} steps')

        for p in self.list_of_parameters:
            # network weights have to be updated outside autocast!
            # we are loading parameters on demand instead of loading them upfront. This reduces memory footprint a lot.
            # each set of parameters is only used once on the test set (one image) so run time wise this is almost the
            # same
            self.network.load_state_dict(torch.load(p, map_location=compute_device)['network_weights'])
            with torch.autocast(self.device.type, enabled=True):
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    pred = self._internal_maybe_mirror_and_predict(data[sl][None].to(compute_device))[0].to(
                        predicted_logits_device)
                    pred /= (pred.max() / 100)
                    predicted_logits[sl] += (pred * gaussian)
                del pred
        empty_cache(self.device)
        return predicted_logits

    def convert_predicted_logits_to_segmentation_with_correct_shape(self, predicted_logits, props):
        old = torch.get_num_threads()
        torch.set_num_threads(7)

        # resample to original shape
        spacing_transposed = [props['spacing'][i] for i in self.plans_manager.transpose_forward]
        current_spacing = self.configuration_manager.spacing if \
            len(self.configuration_manager.spacing) == \
            len(props['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *self.configuration_manager.spacing]
        predicted_logits = self.configuration_manager.resampling_fn_probabilities(predicted_logits,
                                                                                  props[
                                                                                      'shape_after_cropping_and_before_resampling'],
                                                                                  current_spacing,
                                                                                  [props['spacing'][i] for i in
                                                                                   self.plans_manager.transpose_forward])

        segmentation = None
        pp = None
        try:
            with torch.no_grad():
                pp = predicted_logits.to('cuda:0')
                segmentation = pp.argmax(0).cpu()
                del pp
        except RuntimeError:
            del segmentation, pp
            torch.cuda.empty_cache()
            segmentation = predicted_logits.argmax(0)
        del predicted_logits

        # segmentation may be torch.Tensor but we continue with numpy
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()

        # put segmentation in bbox (revert cropping)
        segmentation_reverted_cropping = np.zeros(props['shape_before_cropping'],
                                                  dtype=np.uint8 if len(
                                                      self.label_manager.foreground_labels) < 255 else np.uint16)
        slicer = bounding_box_to_slice(props['bbox_used_for_cropping'])
        segmentation_reverted_cropping[slicer] = segmentation
        del segmentation

        # revert transpose
        segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(self.plans_manager.transpose_backward)
        torch.set_num_threads(old)
        return segmentation_reverted_cropping


def predict_semseg(im, prop, semseg_trained_model, semseg_folds):
    # initialize predictors
    pred_semseg = CustomPredictor(
        tile_step_size=0.5,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=False,
        allow_tqdm=True
    )
    pred_semseg.initialize_from_trained_model_folder(
        semseg_trained_model,
        use_folds=semseg_folds,
        checkpoint_name='checkpoint_final.pth'
    )

    semseg_pred = pred_semseg.predict_single_npy_array(
        im, prop, None
    )
    torch.cuda.empty_cache()
    gc.collect()
    return semseg_pred


def map_labels_to_toothfairy(predicted_seg: np.ndarray) -> np.ndarray:
    # Create an array that maps the labels directly
    max_label = 42
    mapping = np.arange(max_label + 1)

    # Define the specific remapping
    remapping = {19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28,
                 27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,
                 35: 41, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48}

    # Apply the remapping
    for k, v in remapping.items():
        mapping[k] = v

    return mapping[predicted_seg]


def postprocess(prediction_npy, vol_per_voxel, verbose: bool = False):
    cutoffs = {1: 0.0,
               2: 78411.5,
               3: 0.0,
               4: 0.0,
               5: 2800.0,
               6: 1216.5,
               7: 0.0,
               8: 6222.0,
               9: 1573.0,
               10: 946.0,
               11: 0.0,
               12: 6783.5,
               13: 9469.5,
               14: 0.0,
               15: 2260.0,
               16: 3566.0,
               17: 6321.0,
               18: 4221.5,
               19: 5829.0,
               20: 0.0,
               21: 0.0,
               22: 468.0,
               23: 1555.0,
               24: 1291.5,
               25: 2834.5,
               26: 584.5,
               27: 0.0,
               28: 0.0,
               29: 0.0,
               30: 0.0,
               31: 1935.5,
               32: 0.0,
               33: 0.0,
               34: 6140.0,
               35: 0.0,
               36: 0.0,
               37: 0.0,
               38: 2710.0,
               39: 0.0,
               40: 0.0,
               41: 0.0,
               42: 970.0}

    vol_per_voxel_cutoffs = 0.3 * 0.3 * 0.3
    for c in cutoffs.keys():
        co = cutoffs[c]
        if co > 0:
            mask = prediction_npy == c
            pred_vol = np.sum(mask) * vol_per_voxel
            if 0 < pred_vol < (co * vol_per_voxel_cutoffs):
                prediction_npy[mask] = 0
                if verbose:
                    print(
                        f'removed label {c} because predicted volume of {pred_vol} is less than the cutoff {co * vol_per_voxel_cutoffs}')
    return prediction_npy


if __name__ == '__main__':
    os.environ['nnUNet_compile'] = 'f'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=Path, default="/input/images/cbct/")
    parser.add_argument('-o', '--output_folder', type=Path, default="/output/images/oral-pharyngeal-segmentation/")
    parser.add_argument('-sem_mod', '--semseg_trained_model', type=str,
                        default="/opt/app/_trained_model/semseg_trained_model")
    parser.add_argument('--semseg_folds', type=str, nargs='+', default=[0, 1])
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)

    semseg_folds = [i if i == 'all' else int(i) for i in args.semseg_folds]
    semseg_trained_model = args.semseg_trained_model

    rw = SimpleITKIO()

    input_files = list(args.input_folder.glob('*.nii.gz')) + list(args.input_folder.glob('*.mha'))

    for input_fname in input_files:
        output_fname = args.output_folder / input_fname.name

        # we start with the instance seg because we can then start converting that while semseg is being predicted
        # load test image
        im, prop = rw.read_images([input_fname])

        with torch.no_grad():
            semseg_pred = predict_semseg(im, prop, semseg_trained_model, semseg_folds)
            torch.cuda.empty_cache()
            gc.collect()

        # now postprocess
        semseg_pred = postprocess(semseg_pred, np.prod(prop['spacing']), True)

        semseg_pred = map_labels_to_toothfairy(semseg_pred)

        # now save
        rw.write_seg(semseg_pred, output_fname, prop)
