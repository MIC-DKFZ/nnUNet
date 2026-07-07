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
from nnunetv2.utilities.load_weights_utils import *
warmup_stages = Literal["warmup_all", "warmup_decoder", "train_all", "train_decoder"]
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer

class DynamicPretrainedTrainer(PretrainedTrainer):
    """
    A trainer that dynamically adapts pretrained encoder weights to match
    the target architecture from nnUNet's dynamic planning.

    Handles three cases:
    1. Kernel size mismatch: Averages weights along dimensions where kernel changes
       (e.g., [3,3,3] -> [1,3,3] averages along the first spatial dimension)
    2. Pretrained encoder too deep: Cuts away blocks that are too deep
    3. Target encoder deeper than pretrained: Keeps random init for deeper blocks
    """

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            use_pretrained_weights: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, use_pretrained_weights, device)

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            self.network = self.build_network_architecture(
                architecture_class_name=self.configuration_manager.network_arch_class_name,
                arch_init_kwargs=self.configuration_manager.network_arch_init_kwargs,
                arch_init_kwargs_req_import=self.configuration_manager.network_arch_init_kwargs_req_import,
                input_patch_size=self.configuration_manager.patch_size,
                num_input_channels=self.num_input_channels,
                num_output_channels=self.label_manager.num_segmentation_heads,
                enable_deep_supervision=False,
            ).to(self.device)

            if self.use_pretrained_weights:
                assert (
                        "checkpoint_path" in self.adaptation_info
                ), "`checkpoint_path` not found in plans! Can't load weights"
                assert isfile(
                    self.adaptation_info["checkpoint_path"]
                ), f"Pretrained weights path {self.adaptation_info['checkpoint_path']} does not exist!"

                # Get target architecture parameters from configuration
                target_kernel_sizes = self.configuration_manager.network_arch_init_kwargs.get(
                    "kernel_sizes", [[3, 3, 3]] * 6
                )
                target_n_stages = self.configuration_manager.network_arch_init_kwargs.get("n_stages", 6)

                self.network, self.pt_weight_in_ch_mismatch = self.load_pretrained_weights_dynamic(
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
                    target_kernel_sizes=target_kernel_sizes,
                    target_n_stages=target_n_stages,
                )
                self.print_citations()
                self.print_to_log_file(
                    "Loaded Network (with dynamic adaptation) from {}".format(self.adaptation_info["checkpoint_path"])
                )
            else:
                self.print_to_log_file("You are using a Trainer for fine-tuning but without loading weights")

            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = self.network.to(self.device)
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(
                    self.network, device_ids=[self.local_rank], find_unused_parameters=True, static_graph=False
                )
            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized.")

    @staticmethod
    def load_pretrained_weights_dynamic(
            network: AbstractDynamicNetworkArchitectures,
            pretrained_weights_path: str,
            pt_input_channels: int,
            downstream_input_channels: int,
            pt_input_patchsize: int,
            downstream_input_patchsize: int,
            pt_key_to_encoder: str,
            pt_key_to_stem: str,
            pt_keys_to_in_proj: tuple[str, ...],
            pt_key_to_lpe: str,
            target_kernel_sizes: List[List[int]],
            target_n_stages: int,
    ) -> tuple[nn.Module, bool]:
        """
        Load pretrained weights with dynamic adaptation for architecture mismatches.

        This method extends the base load_pretrained_weights to handle:
        1. Kernel size mismatches between pretrained and target (averages weights)
        2. Depth mismatches (cuts or leaves random init as appropriate)

        Args:
            network: The target network to load weights into
            pretrained_weights_path: Path to pretrained checkpoint
            pt_input_channels: Number of input channels in pretrained model
            downstream_input_channels: Number of input channels in target
            pt_input_patchsize: Patch size used in pretraining
            downstream_input_patchsize: Patch size for target
            pt_key_to_encoder: Key path to encoder in pretrained weights
            pt_key_to_stem: Key path to stem in pretrained weights
            pt_keys_to_in_proj: Keys to input projection layers
            pt_key_to_lpe: Key to learnable positional embedding (if any)
            target_kernel_sizes: Kernel sizes per stage in target architecture
            target_n_stages: Number of stages in target architecture

        Returns:
            Tuple of (network with loaded weights, whether channel mismatch occurred)
        """
        from nnunetv2.utilities.load_weights_utils import (
            adapt_encoder_weights_for_architecture,
            handle_pos_embed_resize,
            filter_state_dict,
        )

        key_to_encoder = network.key_to_encoder
        key_to_stem = network.key_to_stem

        ckp = torch.load(pretrained_weights_path, weights_only=True)
        pre_train_statedict: dict[str, torch.Tensor] = ckp["network_weights"]

        # Extract adaptation plan from checkpoint if available
        if "key_to_stem" in ckp.get("nnssl_adaptation_plan", {}).keys():
            pt_key_to_stem = ckp["nnssl_adaptation_plan"]["key_to_stem"]
        if "key_to_encoder" in ckp.get("nnssl_adaptation_plan", {}).keys():
            pt_key_to_encoder = ckp["nnssl_adaptation_plan"]["key_to_encoder"]
        if "keys_to_in_proj" in ckp.get("nnssl_adaptation_plan", {}).keys():
            pt_keys_to_in_proj = ckp["nnssl_adaptation_plan"]["keys_to_in_proj"]
        if "key_to_lpe" in ckp.get("nnssl_adaptation_plan", {}).keys():
            pt_key_to_lpe = ckp["nnssl_adaptation_plan"]["key_to_lpe"]
        if "pretrain_patch_size" in ckp.get("nnssl_adaptation_plan", {}).keys():
            pt_input_patchsize = ckp["nnssl_adaptation_plan"]["pretrain_patch_size"]

        # Extract pretrained architecture info
        pt_arch_kwargs = ckp.get("nnssl_adaptation_plan", {}).get("architecture_kwargs", {})
        pt_kernel_sizes = pt_arch_kwargs.get("kernel_sizes", [[3, 3, 3]] * 6)
        pt_n_stages = pt_arch_kwargs.get("n_stages", 6)

        # If architecture info not in checkpoint, try to infer from weights
        if "architecture_kwargs" not in ckp.get("nnssl_adaptation_plan", {}):
            # Count stages by examining encoder weight keys
            encoder_keys = [k for k in pre_train_statedict.keys() if k.startswith(pt_key_to_encoder)]
            stage_indices = set()
            for k in encoder_keys:
                stripped = k.replace(pt_key_to_encoder + ".", "")
                if stripped[0].isdigit():
                    stage_idx = int(stripped.split(".")[0])
                    stage_indices.add(stage_idx)
            if stage_indices:
                pt_n_stages = max(stage_indices) + 1
            print(f"[Dynamic Adaptation] Inferred pretrained encoder has {pt_n_stages} stages")
            # Assume standard ResEnc-L kernel sizes if not specified
            pt_kernel_sizes = [[3, 3, 3]] * pt_n_stages

        stem_in_encoder = pt_key_to_stem in pre_train_statedict

        pt_weight_in_ch_mismatch = False
        need_to_adapt_lpe = False
        key_to_lpe = getattr(network, "key_to_lpe", None)
        lpe_in_stem = False

        if key_to_lpe is not None:
            lpe_in_encoder = key_to_lpe.startswith(key_to_encoder)
            lpe_in_stem = key_to_lpe.startswith(key_to_stem)
            if pt_input_patchsize != downstream_input_patchsize:
                need_to_adapt_lpe = True

        def strip_dot_prefix(s) -> str:
            if s.startswith("."):
                return s[1:]
            return s

        # Get target state dicts for shape reference
        target_encoder_state = network.get_submodule(key_to_encoder).state_dict()
        target_stem_state = network.get_submodule(key_to_stem).state_dict() if not stem_in_encoder else {}

        if stem_in_encoder:
            encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}

            # Handle input channel mismatch
            if downstream_input_channels > pt_input_channels:
                pt_weight_in_ch_mismatch = True
                k_proj = pt_keys_to_in_proj[0] + ".weight"
                vals = (
                           encoder_weights[k_proj].repeat(1, downstream_input_channels, 1, 1)
                       ) / downstream_input_channels
                for k in pt_keys_to_in_proj:
                    encoder_weights[k] = vals

            # Strip prefix from keys
            new_encoder_weights = {
                strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v for k, v in encoder_weights.items()
            }

            # Apply dynamic architecture adaptation
            new_encoder_weights, adaptation_log = adapt_encoder_weights_for_architecture(
                pretrained_encoder_weights=new_encoder_weights,
                target_state_dict=target_encoder_state,
                pretrained_kernel_sizes=pt_kernel_sizes,
                target_kernel_sizes=target_kernel_sizes,
                pretrained_n_stages=pt_n_stages,
                target_n_stages=target_n_stages,
                verbose=True,
            )

            # Handle LPE adaptation
            if need_to_adapt_lpe:
                if lpe_in_encoder and "pos_embed" in new_encoder_weights:
                    handle_pos_embed_resize(
                        new_encoder_weights,
                        network.get_submodule(key_to_encoder).state_dict(),
                        "interpolate_trilinear",
                        downstream_input_patchsize,
                        pt_input_patchsize,
                        new_encoder_weights.get("down_projection.proj.weight", torch.zeros(1, 1, 1, 1, 1)).shape[2:],
                    )
                    new_encoder_weights["pos_embed"] = new_encoder_weights["pos_embed"].to(
                        next(network.parameters()).device
                    )

            if "cls_token" in new_encoder_weights:
                skip_strings_in_pretrained = ["cls_token"]
                new_encoder_weights, _ = filter_state_dict(new_encoder_weights, skip_strings_in_pretrained)

            # Load with strict=False to allow partial loading
            encoder_module = network.get_submodule(key_to_encoder)
            encoder_module.load_state_dict(new_encoder_weights, strict=False)

        else:
            # Separate stem and encoder
            encoder_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_encoder)}
            stem_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_stem)}

            # Handle input channel mismatch in stem
            if downstream_input_channels > pt_input_channels:
                pt_weight_in_ch_mismatch = True
                k_proj = pt_keys_to_in_proj[0] + ".weight"
                vals = (
                           stem_weights[k_proj].repeat(1, downstream_input_channels, 1, 1, 1)
                       ) / downstream_input_channels
                for k in pt_keys_to_in_proj:
                    stem_weights[k + ".weight"] = vals

            # Strip prefix from keys
            new_encoder_weights = {
                strip_dot_prefix(k.replace(pt_key_to_encoder, "")): v for k, v in encoder_weights.items()
            }
            new_stem_weights = {strip_dot_prefix(k.replace(pt_key_to_stem, "")): v for k, v in stem_weights.items()}

            # Apply dynamic architecture adaptation to encoder
            new_encoder_weights, adaptation_log = adapt_encoder_weights_for_architecture(
                pretrained_encoder_weights=new_encoder_weights,
                target_state_dict=target_encoder_state,
                pretrained_kernel_sizes=pt_kernel_sizes,
                target_kernel_sizes=target_kernel_sizes,
                pretrained_n_stages=pt_n_stages,
                target_n_stages=target_n_stages,
                verbose=True,
            )

            # Handle stem kernel adaptation if needed
            # Compare pretrained stem weights against target stem state dict
            from nnunetv2.utilities.load_weights_utils import adapt_conv_kernel_size

            target_stem_state = network.get_submodule(key_to_stem).state_dict()

            for key in list(new_stem_weights.keys()):
                if key not in target_stem_state:
                    continue
                pt_weight = new_stem_weights[key]
                target_weight = target_stem_state[key]

                # Check if this is a conv weight that needs kernel adaptation
                if ("conv.weight" in key or (key.endswith(".weight") and "conv" in key)) and len(
                        pt_weight.shape
                ) >= 4:
                    pt_k = list(pt_weight.shape[2:])
                    target_k = list(target_weight.shape[2:])

                    if pt_k != target_k:
                        print(f"[Dynamic Adaptation] Adapting stem kernel for {key}: {pt_k} -> {target_k}")
                        new_stem_weights[key] = adapt_conv_kernel_size(pt_weight, target_k, pt_k)

            # Handle LPE adaptation
            if need_to_adapt_lpe:
                if lpe_in_stem and "pos_embed" in new_stem_weights:
                    handle_pos_embed_resize(
                        new_stem_weights,
                        network.get_submodule(key_to_stem).state_dict(),
                        "interpolate_trilinear",
                        downstream_input_patchsize,
                        pt_input_patchsize,
                        new_stem_weights.get("proj.weight", torch.zeros(1, 1, 1, 1, 1)).shape[2:],
                    )
                    new_stem_weights["pos_embed"] = new_stem_weights["pos_embed"].to(
                        next(network.parameters()).device
                    )
                elif lpe_in_encoder and "pos_embed" in new_encoder_weights:
                    handle_pos_embed_resize(
                        new_encoder_weights,
                        network.get_submodule(key_to_encoder).state_dict(),
                        "interpolate_trilinear",
                        downstream_input_patchsize,
                        pt_input_patchsize,
                        new_stem_weights.get("proj.weight", torch.zeros(1, 1, 1, 1, 1)).shape[2:],
                    )
                    new_encoder_weights["pos_embed"] = new_encoder_weights["pos_embed"].to(
                        next(network.parameters()).device
                    )

            if "cls_token" in new_encoder_weights:
                skip_strings_in_pretrained = ["cls_token"]
                new_encoder_weights, _ = filter_state_dict(new_encoder_weights, skip_strings_in_pretrained)

            # Load with strict=False to allow partial loading
            encoder_module = network.get_submodule(key_to_encoder)
            encoder_module.load_state_dict(new_encoder_weights, strict=False)

            stem_module = network.get_submodule(key_to_stem)
            stem_module.load_state_dict(new_stem_weights, strict=False)

            del new_stem_weights, stem_weights

        # Handle LPE that's not in encoder or stem
        if not need_to_adapt_lpe and key_to_lpe is not None:
            lpe_weights = {k: v for k, v in pre_train_statedict.items() if k.startswith(pt_key_to_lpe)}
            if len(lpe_weights) == 1:
                network.get_parameter(key_to_lpe).data = list(lpe_weights.values())[0].to(
                    next(network.parameters()).device
                )

        del pre_train_statedict, encoder_weights, new_encoder_weights
        return network, pt_weight_in_ch_mismatch

class DynamicPretrainedTrainer_adam_150ep(DynamicPretrainedTrainer):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            use_pretrained_weights: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, use_pretrained_weights, device)
        self.initial_lr = 1e-4
        if not self.use_pretrained_weights:
            self.initial_lr = 3e-4
        self.num_epochs = 150
        self.warmup_duration_whole_net = 15  # lin increase whole network


    def configure_optimizers(self, stage: str = "warmup_all"):
        assert stage in ["warmup_all", "train"]

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == "warmup_all":
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr, weight_decay=self.weight_decay, amsgrad=False, betas=(0.9, 0.98), fused=True
            )
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == "warmup_all":
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98),
                    fused=True,
                )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
            )
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

class DynamicPretrainedTrainer_nomirroring(DynamicPretrainedTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

class DynamicPretrainedTrainer_adam_150ep_nomirroring(DynamicPretrainedTrainer_adam_150ep):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

