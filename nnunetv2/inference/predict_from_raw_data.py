import os
import traceback
from multiprocessing import Pool
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction
from nnunetv2.inference.sliding_window_prediction import predict_sliding_window_return_logits, compute_gaussian
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.utils import get_preprocessor_class_from_plans
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels, convert_labelmap_to_one_hot
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists: List[List[str]], list_of_segs_from_prev_stage_files: Union[List[None], List[str]],
                 preprocessor: DefaultPreprocessor, output_filenames_truncated: List[str],
                 plans: dict, dataset_json: dict, configuration: str, dataset_fingerprint: dict,
                 num_threads_in_multithreaded: int = 1):
        self.preprocessor, self.plans, self.configuration, self.dataset_json, self.dataset_fingerprint = \
            preprocessor, plans, configuration, dataset_json, dataset_fingerprint

        self.label_manager = get_labelmanager(plans, dataset_json)

        super().__init__(list(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)),
                         1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        files = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        ofile = self._data[idx][2]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg, data_properites = self.preprocessor.run_case(files, seg_prev_stage, self.plans, self.configuration,
                                                              self.dataset_json, self.dataset_fingerprint)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        if np.prod(data.shape) > (2e9 / 4 * 0.85):
            # we need to temporarily save the preprocessed image due to process-process communication restrictions
            np.save(ofile + '.npy', data)
            data = ofile + '.npy'

        return {'data': data, 'data_properites': data_properites, 'ofile': ofile}


def predict_from_raw_data(list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          output_folder: str,
                          model_training_output_dir: str,
                          use_folds: Union[Tuple[int], str] = None,
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
                          part_id: int = 0):
    # we could also load plans and dataset_json from the init arguments in the checkpoint but then would still have to
    # load the fingerprint from file. Not quite sure what is the best method so we leave things as they are for the
    # moment.
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    dataset_fingerprint = load_json(join(model_training_output_dir, 'dataset_fingerprint.json'))

    label_manager = get_labelmanager(plans, dataset_json)

    maybe_mkdir_p(output_folder)

    if use_folds is None:
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(output_folder, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(output_folder, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')

    # load parameters
    parameters = []
    if isinstance(use_folds, str):
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{use_folds}', checkpoint_name),
                                map_location=torch.device('cpu'))
        configuration = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
            'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])
    else:
        for i, f in enumerate(use_folds):
            f = int(f)
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                configuration = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])

    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]
    print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')

    output_filename_truncated = [join(output_folder, i) for i in caseids]
    seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending']) if
                                 folder_with_segs_from_prev_stage is not None else None for i in caseids]

    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
        seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
        print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
              f'That\'s {len(not_existing_indices)} cases.')
        # caseids = [caseids[i] for i in not_existing_indices]

    # we need to somehow get the configuration. We could do this via the path but I think this is not ideal. Maybe we
    # need to save an extra file?
    preprocessor = get_preprocessor_class_from_plans(plans, configuration)()

    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = max(1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder)))
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, seg_from_prev_stage_files, preprocessor,
                            output_filename_truncated, plans, dataset_json, configuration, dataset_fingerprint,
                            num_processes)
    mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=True)
    # mta = SingleThreadedAugmenter(ppa, NumpyToTensor())

    # restore network
    num_input_channels = determine_num_input_channels(plans, configuration, dataset_json)
    network = get_network_from_plans(plans, dataset_json, configuration, num_input_channels, deep_supervision=True)
    network.decoder.deep_supervision = False
    if torch.cuda.is_available():
        network = network.to('cuda:0')

    # precompute gaussian
    inference_gaussian = torch.from_numpy(compute_gaussian(plans['configurations'][configuration]['patch_size']))
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to('cuda:0')

    num_seg_heads = label_manager.num_segmentation_heads

    # go go go
    export_pool = Pool(num_processes_segmentation_export)
    r = []
    with torch.no_grad():
        for preprocessed in mta:
            data = preprocessed['data']
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed['ofile']
            print(f'\nPredicting {os.path.basename(ofile)}:')
            print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')

            properties = preprocessed['data_properites']

            # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
            # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
            # things a lot faster for some datasets.
            prediction = None
            overwrite_perform_everything_on_gpu = perform_everything_on_gpu
            if perform_everything_on_gpu:
                try:
                    for params in parameters:
                        network.load_state_dict(params)
                        if prediction is None:
                            prediction = predict_sliding_window_return_logits(
                                network, data, num_seg_heads,
                                plans['configurations'][configuration]['patch_size'],
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose)
                        else:
                            prediction += predict_sliding_window_return_logits(
                                network, data, num_seg_heads,
                                plans['configurations'][configuration]['patch_size'],
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose)
                        if len(parameters) > 1:
                            prediction /= len(parameters)

                except RuntimeError:
                    print('Predicton with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                          'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                    print('Error:')
                    traceback.print_exc()
                    prediction = None
                    overwrite_perform_everything_on_gpu = False

            if prediction is None:
                for params in parameters:
                    network.load_state_dict(params)
                    if prediction is None:
                        prediction = predict_sliding_window_return_logits(
                            network, data, num_seg_heads,
                            plans['configurations'][configuration]['patch_size'],
                            mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                            tile_step_size=tile_step_size,
                            use_gaussian=use_gaussian,
                            precomputed_gaussian=inference_gaussian,
                            perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                            verbose=verbose)
                    else:
                        prediction += predict_sliding_window_return_logits(
                            network, data, num_seg_heads,
                            plans['configurations'][configuration]['patch_size'],
                            mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                            tile_step_size=tile_step_size,
                            use_gaussian=use_gaussian,
                            precomputed_gaussian=inference_gaussian,
                            perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                            verbose=verbose)
                    if len(parameters) > 1:
                        prediction /= len(parameters)

            print('Prediction done, transferring to CPU if needed')
            prediction = prediction.to('cpu').numpy()

            if np.prod(prediction.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                print('output is too large for python process-process communication. Saving to file...')
                np.save(ofile + '.npy', prediction)
                prediction = ofile + '.npy'

            # this needs to go into background processes
            # export_prediction(prediction, properties, configuration, plans, dataset_json, ofile,
            #                   save_probabilities)
            print('sending off prediction to background worker for resampling and export')
            r.append(
                export_pool.starmap_async(
                    export_prediction, ((prediction, properties, configuration, plans, dataset_json, ofile,
                                         save_probabilities),)
                )
            )
            print(f'done with {os.path.basename(ofile)}')
    [i.get() for i in r]
    export_pool.close()
    export_pool.join()


def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct suffixes for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=int, required=False, default=(0, 1, 2, 3, 4),
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
    args = parser.parse_args()

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

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
                          folder_with_segs_from_prev_stage=args.prev_stage_predictions)


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct suffixes for your files (_0000 etc). '
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
    parser.add_argument('-f', nargs='+', type=int, required=False, default=(0, 1, 2, 3, 4),
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


    args = parser.parse_args()

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive agressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    predict_from_raw_data(args.i,
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
                          part_id=args.part_id)


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
