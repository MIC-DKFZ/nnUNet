import inspect
import multiprocessing
import os
import shutil
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import get_data_iterator_from_lists_of_filenames
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import predict_sliding_window_return_logits, compute_gaussian
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def load_trained_model_for_inference(model_training_output_dir, use_folds, checkpoint_name):
    # we could also load plans and dataset_json from the init arguments in the checkpoint. Not quite sure what is the
    # best method so we leave things as they are for the moment.
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
    network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                       num_input_channels, enable_deep_supervision=False)
    return parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network, trainer_name


def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
    print('use_folds is None, attempting to auto detect available folds')
    fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
    fold_folders = [i for i in fold_folders if i != 'fold_all']
    fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
    use_folds = [int(i.split('_')[-1]) for i in fold_folders]
    print(f'found the following folds: {use_folds}')
    return use_folds


def manage_input_and_output_lists(list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                  output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                  dataset_json: dict,
                                  folder_with_segs_from_prev_stage: str = None,
                                  overwrite: bool = True,
                                  part_id: int = 0, num_parts: int = 1,
                                  save_probabilities: bool = False):
    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])
    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]
    print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')

    if isinstance(output_folder_or_list_of_truncated_output_files, str):
        output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
    else:
        output_filename_truncated = output_folder_or_list_of_truncated_output_files

    seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending']) if
                                 folder_with_segs_from_prev_stage is not None else None for i in caseids]
    # remove already predicted files form the lists
    if not overwrite and output_filename_truncated is not None:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        if save_probabilities:
            tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
            tmp = [i and j for i, j in zip(tmp, tmp2)]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
        seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
        print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
              f'That\'s {len(not_existing_indices)} cases.')
    return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files


def predict_from_raw_data(list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                          model_training_output_dir: str,
                          use_folds: Union[Tuple[int, ...], str] = None,
                          tile_step_size: float = 0.5,
                          use_gaussian: bool = True,
                          use_mirroring: bool = True,
                          perform_everything_on_gpu: bool = True,
                          verbose: bool = True,
                          save_probabilities: bool = False,
                          overwrite: bool = True,
                          checkpoint_name: str = 'checkpoint_final.pth',
                          num_processes_preprocessing: int = default_num_processes,
                          num_processes_segmentation_export: int = default_num_processes,
                          folder_with_segs_from_prev_stage: str = None,
                          num_parts: int = 1,
                          part_id: int = 0,
                          device: torch.device = torch.device('cuda')):
    """
    This is nnU-Net's default function for making predictions. It works best for batch predictions
    (predicting many images at once).
    """
    print("\n#######################################################################\nPlease cite the following paper "
          "when using nnU-Net:\n"
          "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
          "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
          "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    if device.type == 'cuda':
        device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!

    if device.type != 'cuda':
        perform_everything_on_gpu = False

    ########################
    # let's store the input arguments so that its clear what was used to generate the prediction
    if isinstance(output_folder_or_list_of_truncated_output_files, str):
        output_folder = output_folder_or_list_of_truncated_output_files
    elif isinstance(output_folder_or_list_of_truncated_output_files, list):
        output_folder = os.path.basename(output_folder_or_list_of_truncated_output_files[0])
    else:
        output_folder = None
    if output_folder is not None:
        my_init_kwargs = {}
        for k in inspect.signature(predict_from_raw_data).parameters.keys():
            my_init_kwargs[k] = locals()[k]
        my_init_kwargs = deepcopy(my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
        recursive_fix_for_json_export(my_init_kwargs)
        maybe_mkdir_p(output_folder)
        save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

        # we need these two if we want to do things with the predictions like for example apply postprocessing
        shutil.copy(join(model_training_output_dir, 'dataset.json'), join(output_folder, 'dataset.json'))
        shutil.copy(join(model_training_output_dir, 'plans.json'), join(output_folder, 'plans.json'))
    #######################

    if use_folds is None:
        use_folds = auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    # load all the stuff we need from the model_training_output_dir
    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        load_trained_model_for_inference(model_training_output_dir, use_folds, checkpoint_name)

    # check if we need a prediction from the previous stage
    if configuration_manager.previous_stage_name is not None:
        if folder_with_segs_from_prev_stage is None:
            print(f'WARNING: The requested configuration is a cascaded model and requires predctions from the '
                  f'previous stage! folder_with_segs_from_prev_stage was not provided. Trying to run the '
                  f'inference of the previous stage...')
            assert output_folder is not None, "This does not work if we are not given a folder or a list of strings " \
                                              "for output_folder_or_list_of_output_files"
            folder_with_segs_from_prev_stage = join(output_folder,
                                                    f'prediction_{configuration_manager.previous_stage_name}')
            # we can only do this if we do not have multiple parts
            assert num_parts == 1 and part_id == 0, "folder_with_segs_from_prev_stage was not given and inference " \
                                                    "is distributed over more than one part (num_parts > 1). Cannot " \
                                                    "automatically run predictions for the previous stage"
            predict_from_raw_data(list_of_lists_or_source_folder,
                                  folder_with_segs_from_prev_stage,
                                  get_output_folder(plans_manager.dataset_name,
                                                    trainer_name,
                                                    plans_manager.plans_name,
                                                    configuration_manager.previous_stage_name),
                                  use_folds, tile_step_size, use_gaussian, use_mirroring, perform_everything_on_gpu,
                                  verbose, False, overwrite, checkpoint_name,
                                  num_processes_preprocessing, num_processes_segmentation_export, None,
                                  num_parts=num_parts, part_id=part_id, device=device)

    # sort out input and output filenames
    list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
        manage_input_and_output_lists(list_of_lists_or_source_folder, output_folder_or_list_of_truncated_output_files,
                                      dataset_json, folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                      save_probabilities)

    data_iterator = get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder, seg_from_prev_stage_files,
                                                    output_filename_truncated, configuration_manager, plans_manager, dataset_json,
                                                    num_processes_preprocessing, pin_memory=device.type == 'cuda',
                                                    verbose=verbose)

    if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) and \
            (
                    len(list_of_lists_or_source_folder) > 1):  # just a dumb heurisitic in order to skip compiling for few inference cases
        print('compiling network')
        network = torch.compile(network)

    return predict_from_data_iterator(data_iterator, network, parameters, plans_manager, configuration_manager, dataset_json,
                               inference_allowed_mirroring_axes, tile_step_size, use_gaussian, use_mirroring,
                               perform_everything_on_gpu, verbose, save_probabilities,
                               num_processes_segmentation_export, device)


def predict_from_data_iterator(data_iterator,
                               network: nn.Module,
                               parameter_list: List[dict],
                               plans_manager: PlansManager,
                               configuration_manager: ConfigurationManager,
                               dataset_json: dict,
                               inference_allowed_mirroring_axes: Tuple[int, ...],
                               tile_step_size: float = 0.5,
                               use_gaussian: bool = True,
                               use_mirroring: bool = True,
                               perform_everything_on_gpu: bool = True,
                               verbose: bool = True,
                               save_probabilities: bool = False,
                               num_processes_segmentation_export: int = default_num_processes,
                               device: torch.device = torch.device('cuda')):
    """
    each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
    """
    with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
        network = network.to(device)

        r = []
        with torch.no_grad():
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')
                print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')

                properties = preprocessed['data_properites']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))

                prediction = predict_logits_from_preprocessed_data(data, network, parameter_list, plans_manager,
                                                                   configuration_manager, dataset_json,
                                                                   inference_allowed_mirroring_axes, tile_step_size,
                                                                   use_gaussian, use_mirroring,
                                                                   perform_everything_on_gpu,
                                                                   verbose, device)
                prediction = prediction.numpy()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
                    #                               dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits, ((prediction, properties, configuration_manager, plans_manager,
                                                             dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    label_manager = plans_manager.get_label_manager(dataset_json)
                    # convert_predicted_logits_to_segmentation_with_correct_shape(prediction, plans_manager,
                    #                                                             configuration_manager, label_manager,
                    #                                                             properties,
                    #                                                             save_probabilities)
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, plans_manager,
                                 configuration_manager, label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
        ret = [i.get()[0] for i in r]

    if isinstance(data_iterator, MultiThreadedAugmenter):
        data_iterator._finish()

    # clear lru cache
    compute_gaussian.cache_clear()
    # clear device cache
    empty_cache(device)
    return ret


def predict_logits_from_preprocessed_data(data: torch.Tensor,
                                          network: nn.Module,
                                          parameter_list: List[dict],
                                          plans_manager: PlansManager,
                                          configuration_manager: ConfigurationManager,
                                          dataset_json: dict,
                                          inference_allowed_mirroring_axes: Tuple[int, ...],
                                          tile_step_size: float = 0.5,
                                          use_gaussian: bool = True,
                                          use_mirroring: bool = True,
                                          perform_everything_on_gpu: bool = True,
                                          verbose: bool = True,
                                          device: torch.device = torch.device('cuda')
                                          ) -> torch.Tensor:
    """
    IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
    TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!
    """
    # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
    # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
    # things a lot faster for some datasets.
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    prediction = None
    overwrite_perform_everything_on_gpu = perform_everything_on_gpu
    if perform_everything_on_gpu:
        try:
            for params in parameter_list:
                # messing with state dict names...
                if not isinstance(network, OptimizedModule):
                    network.load_state_dict(params)
                else:
                    network._orig_mod.load_state_dict(params)

                if prediction is None:
                    prediction = predict_sliding_window_return_logits(
                        network, data, num_seg_heads,
                        configuration_manager.patch_size,
                        mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                        tile_step_size=tile_step_size,
                        use_gaussian=use_gaussian,
                        precomputed_gaussian=None,
                        perform_everything_on_gpu=perform_everything_on_gpu,
                        verbose=verbose,
                        device=device)
                else:
                    prediction += predict_sliding_window_return_logits(
                        network, data, num_seg_heads,
                        configuration_manager.patch_size,
                        mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                        tile_step_size=tile_step_size,
                        use_gaussian=use_gaussian,
                        precomputed_gaussian=None,
                        perform_everything_on_gpu=perform_everything_on_gpu,
                        verbose=verbose,
                        device=device)
            if len(parameter_list) > 1:
                prediction /= len(parameter_list)

        except RuntimeError:
            print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                  'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
            print('Error:')
            traceback.print_exc()
            prediction = None
            overwrite_perform_everything_on_gpu = False

    if prediction is None:
        for params in parameter_list:
            # messing with state dict names...
            if not isinstance(network, OptimizedModule):
                network.load_state_dict(params)
            else:
                network._orig_mod.load_state_dict(params)

            if prediction is None:
                prediction = predict_sliding_window_return_logits(
                    network, data, num_seg_heads,
                    configuration_manager.patch_size,
                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    precomputed_gaussian=None,
                    perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                    verbose=verbose,
                    device=device)
            else:
                prediction += predict_sliding_window_return_logits(
                    network, data, num_seg_heads,
                    configuration_manager.patch_size,
                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                    tile_step_size=tile_step_size,
                    use_gaussian=use_gaussian,
                    precomputed_gaussian=None,
                    perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                    verbose=verbose,
                    device=device)
        if len(parameter_list) > 1:
            prediction /= len(parameter_list)

    print('Prediction done, transferring to CPU if needed')
    prediction = prediction.to('cpu')
    return prediction


def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
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

    predict_from_raw_data(args.i,
                          args.o,
                          args.m,
                          args.f,
                          args.step_size,
                          use_gaussian=True,
                          use_mirroring=not args.disable_tta,
                          perform_everything_on_gpu=True,
                          verbose=args.verbose,
                          save_probabilities=args.save_probabilities,
                          overwrite=not args.continue_prediction,
                          checkpoint_name=args.chk,
                          num_processes_preprocessing=args.npp,
                          num_processes_segmentation_export=args.nps,
                          folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                          device=device)


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive agressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
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

    r = predict_from_raw_data(args.i,
                          args.o,
                          model_folder,
                          args.f,
                          args.step_size,
                          use_gaussian=True,
                          use_mirroring=not args.disable_tta,
                          perform_everything_on_gpu=True,
                          verbose=args.verbose,
                          save_probabilities=args.save_probabilities,
                          overwrite=not args.continue_prediction,
                          checkpoint_name=args.chk,
                          num_processes_preprocessing=args.npp,
                          num_processes_segmentation_export=args.nps,
                          folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                          num_parts=args.num_parts,
                          part_id=args.part_id,
                          device=device)


if __name__ == '__main__':
    predict_from_raw_data('/media/fabian/data/nnUNet_raw/Dataset003_Liver/imagesTs',
                          '/media/fabian/data/nnUNet_raw/Dataset003_Liver/imagesTs_predlowres',
                          '/home/fabian/results/nnUNet_remake/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres',
                          (0,),
                          0.5,
                          use_gaussian=True,
                          use_mirroring=False,
                          perform_everything_on_gpu=True,
                          verbose=True,
                          save_probabilities=False,
                          overwrite=False,
                          checkpoint_name='checkpoint_final.pth',
                          num_processes_preprocessing=3,
                          num_processes_segmentation_export=3)

    predict_from_raw_data('/media/fabian/data/nnUNet_raw/Dataset003_Liver/imagesTs',
                          '/media/fabian/data/nnUNet_raw/Dataset003_Liver/imagesTs_predCascade',
                          '/home/fabian/results/nnUNet_remake/Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres',
                          (0,),
                          0.5,
                          use_gaussian=True,
                          use_mirroring=False,
                          perform_everything_on_gpu=True,
                          verbose=True,
                          save_probabilities=False,
                          overwrite=True,
                          checkpoint_name='checkpoint_final.pth',
                          num_processes_preprocessing=2,
                          num_processes_segmentation_export=2,
                          folder_with_segs_from_prev_stage='/media/fabian/data/nnUNet_raw/Dataset003_Liver/imagesTs_predlowres')
