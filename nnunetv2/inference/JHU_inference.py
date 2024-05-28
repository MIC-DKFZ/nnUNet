import argparse
import multiprocessing
import os
from time import sleep
from typing import Union

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle, join, maybe_mkdir_p, subdirs

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def export_prediction_from_logits_singleFiles(
        predicted_array_or_file: Union[np.ndarray, torch.Tensor],
        properties_dict: dict,
        configuration_manager: ConfigurationManager,
        plans_manager: PlansManager,
        dataset_json_dict_or_file: Union[dict, str],
        output_file_truncated: str,
        save_probabilities: bool = False):
    """
    This function generates the output structure expected by the JHU benchmark. We interpret output_file_truncated
    as the output folder. We create 'predictions' subfolders and populate them with the label maps
    """

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    output_folder = join(output_file_truncated, 'predictions')
    maybe_mkdir_p(output_folder)
    label_name_dict = {j: i for i, j in label_manager.label_dict.items()}
    for l in label_manager.foreground_labels:
        label_name = label_name_dict[l]
        rw.write_seg(
            (segmentation_final == l).astype(np.uint8, copy=False),
            join(output_folder, label_name + dataset_json_dict_or_file['file_ending']),
            properties_dict
        )


class JHUPredictor(nnUNetPredictor):
    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        We replace export_prediction_from_logits with export_prediction_from_logits_singleFiles to comply with JHU
        benchmark output format expectations
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
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

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                    #                               self.dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits_singleFiles,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(
                    #             prediction, self.plans_manager,
                    #              self.configuration_manager, self.label_manager,
                    #              properties,
                    #              save_probabilities)

                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
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
        empty_cache(self.device)
        return ret


if __name__ == '__main__':
    # python nnunetv2/inference/JHU_inference.py /home/isensee/Downloads/AbdomenAtlasTest /home/isensee/Downloads/AbdomenAtlasTest_pred -model /home/isensee/temp/JHU/trained_model_ep3850
    # /home/isensee/temp/JHU/trained_model_ep3850
    # /home/isensee/Downloads/AbdomenAtlasTest
    # /home/isensee/Downloads/AbdomenAtlasTest_pred

    os.environ['nnUNet_compile'] = 'f'

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('-model', required=True, type=str)
    parser.add_argument('--disable_tqdm', required=False, action='store_true', default=False)
    args = parser.parse_args()

    predictor = JHUPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=not args.disable_tqdm
    )

    predictor.initialize_from_trained_model_folder(
        args.model,
        ('all', ),
        'checkpoint_latest.pth'
    )

    # we need to create list of list of input files
    input_caseids = subdirs(args.input_dir, join=False)
    input_files = [[join(args.input_dir, i, 'ct.nii.gz')] for i in input_caseids]
    output_folders = [join(args.output_dir, i) for i in input_caseids]

    predictor.predict_from_files(
        input_files,
        output_folders,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=3,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
