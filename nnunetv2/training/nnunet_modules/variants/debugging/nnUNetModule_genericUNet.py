from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from typing import Union

import torch
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, labels_to_list_of_regions
from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from torch import nn

from nnunetv2.training.nnunet_modules.nnUNetModule import nnUNetModule
from nnunetv2.utilities.network_initialization import InitWeights_He


class nnUNetModule_GenericUNet(nnUNetModule):
    def __init__(self, dataset_name_or_id: Union[int, str], plans_name: str, configuration: str, fold: int,
                 unpack_dataset: bool = True, folder_with_segs_from_previous_stage: str = None):
        super().__init__(dataset_name_or_id, plans_name, configuration, fold, unpack_dataset,
                         folder_with_segs_from_previous_stage)
        plans = self.plans
        initial_features = plans["configurations"][configuration]["UNet_base_num_features"]

        dim = len(plans["configurations"][configuration]["conv_kernel_sizes"][0])
        conv_op = convert_dim_to_conv_op(dim)

        strides = plans["configurations"][configuration]["pool_op_kernel_sizes"][1:]

        self.network = Generic_UNet(len(self.dataset_json["modality"]),
                                    initial_features,
                                    len(self.dataset_json["labels"]),
                                    len(strides),
                                    2,
                                    2,
                                    conv_op, get_matching_instancenorm(conv_op), {'eps': 1e-5, 'affine': True}, None,
                                                           None,
                                    nn.LeakyReLU, {'inplace': True}, True, False, lambda x: x, InitWeights_He(1e-2),
                                    strides, plans["configurations"][configuration]["conv_kernel_sizes"], False, True, True)

    def on_predict_start(self) -> None:
        self.inference_gaussian = torch.from_numpy(compute_gaussian(
            self.plans['configurations'][self.configuration]['patch_size'], sigma_scale=1. / 8))
        self.network.do_ds = False
        self.inference_segmentation_export_pool = Pool(self.inference_parameters['n_processes_segmentation_export'])

    def on_predict_end(self) -> None:
        self.inference_gaussian = None
        self.trainer.strategy.barrier("prediction")
        self.network.do_ds = True
        self.inference_segmentation_export_pool.close()
        self.inference_segmentation_export_pool.join()
        self.inference_segmentation_export_pool = None

        compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                  join(self.output_folder, 'validation'),
                                  join(self.output_folder, 'validation', 'summary.json'),
                                  recursive_find_reader_writer_by_name(self.plans["image_reader_writer"])(),
                                  self.dataset_json["file_ending"],
                                  self.regions if self.regions is not None else labels_to_list_of_regions(self.labels),
                                  self.ignore_label)