from copy import deepcopy
from typing import Literal, Tuple, Union, List
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from torch._dynamo import OptimizedModule

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.utilities.get_network_via_name import get_network_from_name
from torch import nn, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer_Primus
from dynamic_network_architectures.architectures.primus import Primus

warmup_stages = Literal["warmup_all", "warmup_decoder", "train_all", "train_decoder"]

class PretrainedTrainer_Primusx(PretrainedTrainer_Primus):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Can be overriden to train same architecture from scratch.
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        if not self.use_pretrained_weights:
            self.initial_lr = 3e-4
        self.adaptation_info = self.plans_manager.plans["pretrain_info"]

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # During `nnUNetv2_preprocess_like_nnssl` we create a new plan that specifies the architecture already.
            #   This plan holds details on how the architecture is supposed to be built.

            self.network = self.build_network_architecture(
                architecture_class_name=self.configuration_manager.network_arch_class_name,
                arch_init_kwargs=self.configuration_manager.network_arch_init_kwargs,
                arch_init_kwargs_req_import=self.configuration_manager.network_arch_init_kwargs_req_import,
                input_patch_size=self.configuration_manager.patch_size,  # Set in plan to pt_recommended_patchsize
                num_input_channels=self.num_input_channels,
                num_output_channels=self.label_manager.num_segmentation_heads,
                enable_deep_supervision=False,
            ).to(self.device)

            # Load pretrained weights
            if self.use_pretrained_weights:
                assert (
                        "checkpoint_path" in self.adaptation_info
                ), "`checkpoint_path` not found in plans! Can't load weights"
                assert isfile(
                    self.adaptation_info["checkpoint_path"]
                ), f"Pretrained weights path {self.adaptation_info['checkpoint_path']} does not exist!"
                self.network,  self.pt_weight_in_ch_mismatch = self.load_pretrained_weights(
                    self.network,
                    pretrained_weights_path=self.adaptation_info["checkpoint_path"],
                    pt_input_channels=self.adaptation_info["pt_num_in_channels"],
                    downstream_input_channels=self.num_input_channels,
                    pt_input_patchsize=self.adaptation_info["pt_used_patchsize"],
                    downstream_input_patchsize=self.configuration_manager.patch_size,
                    pt_key_to_encoder=self.adaptation_info["key_to_encoder"],
                    pt_key_to_stem=self.adaptation_info["key_to_stem"],
                    pt_keys_to_in_proj=tuple(self.adaptation_info["keys_to_in_proj"]),
                    pt_key_to_lpe=self.adaptation_info["key_to_lpe"],
                )
                self.print_citations()
                self.print_to_log_file("Loaded Network from {}".format(self.adaptation_info["checkpoint_path"]))
            else:
                self.print_to_log_file("You are using a Trainer for fine-tuning but without loading weigts")
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = self.network.to(self.device)
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
        input_patch_size: tuple[int, int, int] = None,
        ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        if 'init_values' in arch_init_kwargs:
            if isinstance(arch_init_kwargs['init_values'], list):
                arch_init_kwargs['init_values'] = arch_init_kwargs['init_values'][0]
        model = Primus(
            num_input_channels,
            arch_init_kwargs['embed_dim'],
            arch_init_kwargs['patch_embed_size'],
            num_output_channels,
            arch_init_kwargs['encoder_eva_depth'],
            arch_init_kwargs['encoder_eva_numheads'],
            input_patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=arch_init_kwargs['scale_attn_inner'],
            init_values=arch_init_kwargs['init_values'],
        )
        return model



class PretrainedTrainer_Primusx_150ep(PretrainedTrainer_Primusx):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150 # lin increase whole network

class PretrainedTrainer_Primusx_30ep(PretrainedTrainer_Primusx):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 3  # lin increase whole network
        self.num_epochs = 30 # lin increase whole network

class PretrainedTrainer_Primusx_smallerlr(PretrainedTrainer_Primusx):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-5
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50  # lin increase whole network
        if not self.use_pretrained_weights:
            self.initial_lr = 1e-4
        self.adaptation_info = self.plans_manager.plans["pretrain_info"]

class PretrainedTrainer_Primusx_300ep(PretrainedTrainer_Primusx_150ep):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300

class PretrainedTrainer_Primusx_150ep_lr3e4(PretrainedTrainer_Primusx):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.num_epochs = 150 # lin increase whole network
        self.initial_lr = 3e-4

class PretrainedTrainer_Primusx_150ep_small_debug(PretrainedTrainer_Primusx):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (48, 48, 48)
        plans["configurations"][configuration]["batch_size"] = 2
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Can be overriden to train same architecture from scratch.
        self.warmup_duration_whole_net = 15  # lin increase whole network
        self.num_epochs = 150 # lin increase whole network


class PretrainedTrainer_Primusx_nomirroring(PretrainedTrainer_Primusx):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

class PretrainedTrainer_Primusx_150ep_nomirroring(PretrainedTrainer_Primusx_150ep):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes