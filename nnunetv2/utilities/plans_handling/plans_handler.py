from __future__ import annotations

import warnings

from copy import deepcopy
from functools import lru_cache, partial
from typing import Union, Tuple, List, Type, Callable

import numpy as np
import torch

from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_fn_by_name
import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import load_json, join

from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager_class_from_plans

# see https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from typing import TYPE_CHECKING
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

if TYPE_CHECKING:
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner


class ConfigurationManager(object):
    def __init__(self, configuration_dict: dict):
        self.configuration = configuration_dict

        # backwards compatibility
        if 'architecture' not in self.configuration.keys():
            warnings.warn("Detected old nnU-Net plans format. Attempting to reconstruct network architecture "
                          "parameters. If this fails, rerun nnUNetv2_plan_experiment for your dataset. If you use a "
                          "custom architecture, please downgrade nnU-Net to the version you implemented this "
                          "or update your implementation + plans.")
            # try to build the architecture information from old plans, modify configuration dict to match new standard
            unet_class_name = self.configuration["UNet_class_name"]
            if unet_class_name == "PlainConvUNet":
                network_class_name = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
            elif unet_class_name == 'ResidualEncoderUNet':
                network_class_name = "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet"
            else:
                raise RuntimeError(f'Unknown architecture {unet_class_name}. This conversion only supports '
                                   f'PlainConvUNet and ResidualEncoderUNet')

            n_stages = len(self.configuration["n_conv_per_stage_encoder"])

            dim = len(self.configuration["patch_size"])
            conv_op = convert_dim_to_conv_op(dim)
            instnorm = get_matching_instancenorm(dimension=dim)

            arch_dict = {
                'network_class_name': network_class_name,
                'arch_kwargs': {
                    "n_stages": n_stages,
                    "features_per_stage": [min(self.configuration["UNet_base_num_features"] * 2 ** i,
                                               self.configuration["unet_max_num_features"])
                                           for i in range(n_stages)],
                    "conv_op": conv_op.__module__ + '.' + conv_op.__name__,
                    "kernel_sizes": deepcopy(self.configuration["conv_kernel_sizes"]),
                    "strides": deepcopy(self.configuration["pool_op_kernel_sizes"]),
                    "n_conv_per_stage": deepcopy(self.configuration["n_conv_per_stage_encoder"]),
                    "n_conv_per_stage_decoder": deepcopy(self.configuration["n_conv_per_stage_decoder"]),
                    "conv_bias": True,
                    "norm_op": instnorm.__module__ + '.' + instnorm.__name__,
                    "norm_op_kwargs": {
                        "eps": 1e-05,
                        "affine": True
                    },
                    "dropout_op": None,
                    "dropout_op_kwargs": None,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {
                        "inplace": True
                    }
                },
                # these need to be imported with locate in order to use them:
                # `conv_op = pydoc.locate(architecture_kwargs['conv_op'])`
                "_kw_requires_import": [
                    "conv_op",
                    "norm_op",
                    "dropout_op",
                    "nonlin"
                ]
            }
            del self.configuration["UNet_class_name"], self.configuration["UNet_base_num_features"], \
                self.configuration["n_conv_per_stage_encoder"], self.configuration["n_conv_per_stage_decoder"], \
                self.configuration["num_pool_per_axis"], self.configuration["pool_op_kernel_sizes"],\
                self.configuration["conv_kernel_sizes"], self.configuration["unet_max_num_features"]
            self.configuration["architecture"] = arch_dict

    def __repr__(self):
        return self.configuration.__repr__()

    @property
    def data_identifier(self) -> str:
        return self.configuration['data_identifier']

    @property
    def preprocessor_name(self) -> str:
        return self.configuration['preprocessor_name']

    @property
    @lru_cache(maxsize=1)
    def preprocessor_class(self) -> Type[DefaultPreprocessor]:
        preprocessor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing"),
                                                         self.preprocessor_name,
                                                         current_module="nnunetv2.preprocessing")
        return preprocessor_class

    @property
    def batch_size(self) -> int:
        return self.configuration['batch_size']

    @property
    def patch_size(self) -> List[int]:
        return self.configuration['patch_size']

    @property
    def median_image_size_in_voxels(self) -> List[int]:
        return self.configuration['median_image_size_in_voxels']

    @property
    def spacing(self) -> List[float]:
        return self.configuration['spacing']

    @property
    def normalization_schemes(self) -> List[str]:
        return self.configuration['normalization_schemes']

    @property
    def use_mask_for_norm(self) -> List[bool]:
        return self.configuration['use_mask_for_norm']

    @property
    def network_arch_class_name(self) -> str:
        return self.configuration['architecture']['network_class_name']

    @property
    def network_arch_init_kwargs(self) -> dict:
        return self.configuration['architecture']['arch_kwargs']

    @property
    def network_arch_init_kwargs_req_import(self) -> Union[Tuple[str, ...], List[str]]:
        return self.configuration['architecture']['_kw_requires_import']

    @property
    def pool_op_kernel_sizes(self) -> Tuple[Tuple[int, ...], ...]:
        return self.configuration['architecture']['arch_kwargs']['strides']

    @property
    @lru_cache(maxsize=1)
    def resampling_fn_data(self) -> Callable[
        [Union[torch.Tensor, np.ndarray],
         Union[Tuple[int, ...], List[int], np.ndarray],
         Union[Tuple[float, ...], List[float], np.ndarray],
         Union[Tuple[float, ...], List[float], np.ndarray]
         ],
        Union[torch.Tensor, np.ndarray]]:
        fn = recursive_find_resampling_fn_by_name(self.configuration['resampling_fn_data'])
        fn = partial(fn, **self.configuration['resampling_fn_data_kwargs'])
        return fn

    @property
    @lru_cache(maxsize=1)
    def resampling_fn_probabilities(self) -> Callable[
        [Union[torch.Tensor, np.ndarray],
         Union[Tuple[int, ...], List[int], np.ndarray],
         Union[Tuple[float, ...], List[float], np.ndarray],
         Union[Tuple[float, ...], List[float], np.ndarray]
         ],
        Union[torch.Tensor, np.ndarray]]:
        fn = recursive_find_resampling_fn_by_name(self.configuration['resampling_fn_probabilities'])
        fn = partial(fn, **self.configuration['resampling_fn_probabilities_kwargs'])
        return fn

    @property
    @lru_cache(maxsize=1)
    def resampling_fn_seg(self) -> Callable[
        [Union[torch.Tensor, np.ndarray],
         Union[Tuple[int, ...], List[int], np.ndarray],
         Union[Tuple[float, ...], List[float], np.ndarray],
         Union[Tuple[float, ...], List[float], np.ndarray]
         ],
        Union[torch.Tensor, np.ndarray]]:
        fn = recursive_find_resampling_fn_by_name(self.configuration['resampling_fn_seg'])
        fn = partial(fn, **self.configuration['resampling_fn_seg_kwargs'])
        return fn

    @property
    def batch_dice(self) -> bool:
        return self.configuration['batch_dice']

    @property
    def next_stage_names(self) -> Union[List[str], None]:
        ret = self.configuration.get('next_stage')
        if ret is not None:
            if isinstance(ret, str):
                ret = [ret]
        return ret

    @property
    def previous_stage_name(self) -> Union[str, None]:
        return self.configuration.get('previous_stage')


class PlansManager(object):
    def __init__(self, plans_file_or_dict: Union[str, dict]):
        """
        Why do we need this?
        1) resolve inheritance in configurations
        2) expose otherwise annoying stuff like getting the label manager or IO class from a string
        3) clearly expose the things that are in the plans instead of hiding them in a dict
        4) cache shit

        This class does not prevent you from going wild. You can still use the plans directly if you prefer
        (PlansHandler.plans['key'])
        """
        self.plans = plans_file_or_dict if isinstance(plans_file_or_dict, dict) else load_json(plans_file_or_dict)

    def __repr__(self):
        return self.plans.__repr__()

    def _internal_resolve_configuration_inheritance(self, configuration_name: str,
                                                    visited: Tuple[str, ...] = None) -> dict:
        if configuration_name not in self.plans['configurations'].keys():
            raise ValueError(f'The configuration {configuration_name} does not exist in the plans I have. Valid '
                             f'configuration names are {list(self.plans["configurations"].keys())}.')
        configuration = deepcopy(self.plans['configurations'][configuration_name])
        if 'inherits_from' in configuration:
            parent_config_name = configuration['inherits_from']

            if visited is None:
                visited = (configuration_name,)
            else:
                if parent_config_name in visited:
                    raise RuntimeError(f"Circular dependency detected. The following configurations were visited "
                                       f"while solving inheritance (in that order!): {visited}. "
                                       f"Current configuration: {configuration_name}. Its parent configuration "
                                       f"is {parent_config_name}.")
                visited = (*visited, configuration_name)

            base_config = self._internal_resolve_configuration_inheritance(parent_config_name, visited)
            base_config.update(configuration)
            configuration = base_config
        return configuration

    @lru_cache(maxsize=10)
    def get_configuration(self, configuration_name: str):
        if configuration_name not in self.plans['configurations'].keys():
            raise RuntimeError(f"Requested configuration {configuration_name} not found in plans. "
                               f"Available configurations: {list(self.plans['configurations'].keys())}")

        configuration_dict = self._internal_resolve_configuration_inheritance(configuration_name)
        return ConfigurationManager(configuration_dict)

    @property
    def dataset_name(self) -> str:
        return self.plans['dataset_name']

    @property
    def plans_name(self) -> str:
        return self.plans['plans_name']

    @property
    def original_median_spacing_after_transp(self) -> List[float]:
        return self.plans['original_median_spacing_after_transp']

    @property
    def original_median_shape_after_transp(self) -> List[float]:
        return self.plans['original_median_shape_after_transp']

    @property
    @lru_cache(maxsize=1)
    def image_reader_writer_class(self) -> Type[BaseReaderWriter]:
        return recursive_find_reader_writer_by_name(self.plans['image_reader_writer'])

    @property
    def transpose_forward(self) -> List[int]:
        return self.plans['transpose_forward']

    @property
    def transpose_backward(self) -> List[int]:
        return self.plans['transpose_backward']

    @property
    def available_configurations(self) -> List[str]:
        return list(self.plans['configurations'].keys())

    @property
    @lru_cache(maxsize=1)
    def experiment_planner_class(self) -> Type[ExperimentPlanner]:
        planner_name = self.experiment_planner_name
        experiment_planner = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                         planner_name,
                                                         current_module="nnunetv2.experiment_planning")
        return experiment_planner

    @property
    def experiment_planner_name(self) -> str:
        return self.plans['experiment_planner_used']

    @property
    @lru_cache(maxsize=1)
    def label_manager_class(self) -> Type[LabelManager]:
        return get_labelmanager_class_from_plans(self.plans)

    def get_label_manager(self, dataset_json: dict, **kwargs) -> LabelManager:
        return self.label_manager_class(label_dict=dataset_json['labels'],
                                        regions_class_order=dataset_json.get('regions_class_order'),
                                        **kwargs)

    @property
    def foreground_intensity_properties_per_channel(self) -> dict:
        if 'foreground_intensity_properties_per_channel' not in self.plans.keys():
            if 'foreground_intensity_properties_by_modality' in self.plans.keys():
                return self.plans['foreground_intensity_properties_by_modality']
        return self.plans['foreground_intensity_properties_per_channel']


if __name__ == '__main__':
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    plans = load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(3), 'nnUNetPlans.json'))
    # build new configuration that inherits from 3d_fullres
    plans['configurations']['3d_fullres_bs4'] = {
        'batch_size': 4,
        'inherits_from': '3d_fullres'
    }
    # now get plans and configuration managers
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres_bs4')
    print(configuration_manager)  # look for batch size 4
