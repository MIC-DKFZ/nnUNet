import os
from multiprocessing import Pool
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction
from nnunetv2.inference.sliding_window_prediction import predict_sliding_window_return_logits, compute_gaussian
from nnunetv2.preprocessing.utils import get_preprocessor_class_from_plans
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists, preprocessor, output_filename_truncated, plans, dataset_json, configuration,
                 dataset_fingerprint, num_threads_in_multithreaded=1):
        self.preprocessor, self.plans, self.configuration, self.dataset_json, self.dataset_fingerprint = \
            preprocessor, plans, configuration, dataset_json, dataset_fingerprint

        super().__init__(list(zip(list_of_lists, output_filename_truncated)), 1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        files = self._data[idx][0]
        ofile = self._data[idx][1]
        data, _, data_properites = self.preprocessor.run_case(files, None, self.plans, self.configuration,
                                                              self.dataset_json, self.dataset_fingerprint)
        if np.prod(data.shape) > (2e9 / 4 * 0.85):
            # we need to temporarily save the preprocessed image due to process-process communication restrictions
            np.save(ofile + '.npy', data)
            data = ofile + '.npy'

        return {'data': data, 'data_properites': data_properites, 'ofile': ofile}


def predict_from_raw_data(list_of_lists_or_source_folder: Union[str, List[List[str]]], output_folder: str,
                          model_training_output_dir: str, use_folds: Union[Tuple[int], str] = None,
                          tile_step_size: float = 0.5, use_gaussian: bool = True,
                          use_mirroring: bool = True, perform_everything_on_gpu: bool = False,
                          verbose: bool = True, save_probabilities: bool = False, overwrite: bool = True,
                          checkpoint_name: str = 'checkpoint_final.pth',
                          num_processes_preprocessing: int = default_num_processes,
                          num_processes_segmentation_export: int = default_num_processes):
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    dataset_fingerprint = load_json(join(model_training_output_dir, 'dataset_fingerprint.json'))
    maybe_mkdir_p(output_folder)

    # todo auto detect folds
    if use_folds is None:
        raise NotImplementedError

    # load parameters
    parameters = []
    if isinstance(use_folds, str):
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{use_folds}', checkpoint_name),
                                map_location=torch.device('cpu'))
        configuration = checkpoint['hyper_parameters']['configuration']
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']
        inference_nonlinearity = checkpoint['inference_nonlinearity']

        parameters.append(checkpoint['state_dict'])
        # remove the 'network.' prefix from state_dict
        parameters[-1] = {i[len('network.'):]: j for i, j in parameters[-1].items()}
    else:
        for i, f in enumerate(use_folds):
            f = int(f)
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                configuration = checkpoint['hyper_parameters']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']
                inference_nonlinearity = checkpoint['inference_nonlinearity']

            parameters.append(checkpoint['state_dict'])
            # remove the 'network.' prefix from state_dict
            parameters[-1] = {i[len('network.'):]: j for i, j in parameters[-1].items()}

    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])

    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]

    output_filename_truncated = [join(output_folder, i) for i in caseids]

    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        output_filename_truncated = [output_filename_truncated[i] for i in range(len(output_filename_truncated)) if
                                     not tmp[i]]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in
                                          range(len(list_of_lists_or_source_folder)) if not tmp[i]]
        # caseids = [caseids[i] for i in range(len(caseids)) if not tmp[i]]

    # we need to somehow get the configuration. We could do this via the path but I think this is not ideal. Maybe we
    # need to save an extra file?
    preprocessor = get_preprocessor_class_from_plans(plans, configuration)()

    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = min(num_processes_preprocessing, len(list_of_lists_or_source_folder))
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, preprocessor, output_filename_truncated, plans,
                            dataset_json, configuration, dataset_fingerprint, num_processes)
    mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=True)

    # restore network
    network = get_network_from_plans(plans, dataset_json, configuration, deep_supervision=True)
    network.decoder.deep_supervision = False
    if torch.cuda.is_available():
        network = network.to('cuda:0')

    # precompute gaussian
    inference_gaussian = torch.from_numpy(compute_gaussian(plans['configurations'][configuration]['patch_size']))
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to('cuda:0')

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
            if ofile.endswith('hippocampus_005'):
                import IPython;IPython.embed()
            properties = preprocessed['data_properites']

            prediction = None
            for params in parameters:
                network.load_state_dict(params)
                if prediction is None:
                    predicted_logits = predict_sliding_window_return_logits(
                        network, data, len(dataset_json["labels"]),
                        plans['configurations'][configuration]['patch_size'],
                        mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                        tile_step_size=tile_step_size,
                        use_gaussian=use_gaussian,
                        precomputed_gaussian=inference_gaussian,
                        perform_everything_on_gpu=perform_everything_on_gpu,
                        verbose=verbose)
                else:
                    prediction += predict_sliding_window_return_logits(
                        network, data, len(dataset_json["labels"]),
                        plans['configurations'][configuration]['patch_size'],
                        mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                        tile_step_size=tile_step_size,
                        use_gaussian=use_gaussian,
                        precomputed_gaussian=inference_gaussian,
                        perform_everything_on_gpu=perform_everything_on_gpu,
                        verbose=verbose)
                if len(parameters) > 1:
                    prediction /= len(parameters)

            # apply nonlinearity
            prediction = inference_nonlinearity(predicted_logits)
            prediction = prediction.to('cpu').numpy()

            if np.prod(prediction.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                np.save(ofile + '.npy', prediction)
                prediction = ofile + '.npy'

            # this needs to go into background processes
            # export_prediction(prediction, properties, configuration, plans, dataset_json, ofile,
            #                   save_probabilities)
            r.append(
                export_pool.starmap_async(
                    export_prediction, ((prediction, properties, configuration, plans, dataset_json, ofile,
                                         save_probabilities),)
                )
            )
    [i.get() for i in r]
    export_pool.close()
    export_pool.join()


if __name__ == '__main__':
    predict_from_raw_data('/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/imagesTs',
                          '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/imagesTs_prednnUNetRemake',
                          '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres',
                          (0,),
                          0.5,
                          True,
                          True,
                          False,
                          True,
                          False,
                          True,
                          'checkpoint_final.pth')
