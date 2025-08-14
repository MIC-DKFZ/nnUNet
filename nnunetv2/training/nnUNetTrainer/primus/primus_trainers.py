from abc import abstractmethod
from typing import List, Tuple, Union
import torch
from torch import nn, autocast
from dynamic_network_architectures.architectures.primus import Primus
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.lr_schedule.nnUNetTrainer_warmup import nnUNetTrainer_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.utilities.helpers import empty_cache, dummy_context

######################################################
# See this paper for information on Primus!
# Wald*, T., Roy*, S., Isensee*, F., Ulrich, C., Ziegler, S., Trofimova, D., ... & Maier-Hein, K. (2025). Primus: Enforcing attention usage for 3d medical image segmentation. arXiv preprint arXiv:2503.01835.
# * equal contribution
######################################################

class AbstractPrimus(nnUNetTrainer_warmup):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = False

    @abstractmethod
    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        raise NotImplementedError()

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

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


class nnUNet_Primus_S_Trainer(AbstractPrimus):

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        model = Primus(
            num_input_channels,
            396,
            (8, 8, 8),
            num_output_channels,
            12,
            6,
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_B_Trainer(AbstractPrimus):

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        model = Primus(
            num_input_channels,
            792,
            (8, 8, 8),
            num_output_channels,
            12,
            12,
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_M_Trainer(AbstractPrimus):

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        model = Primus(
            num_input_channels,
            864,
            (8, 8, 8),
            num_output_channels,
            16,
            12,
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class nnUNet_Primus_M_Trainer_BS8(nnUNet_Primus_M_Trainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.configuration_manager.configuration["batch_size"] = 8


class nnUNet_Primus_M_Trainer_BS8_2e4(nnUNet_Primus_M_Trainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 2e-4
        self.configuration_manager.configuration["batch_size"] = 8


class nnUNet_Trainer_BS8(nnUNetTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.configuration_manager.configuration["batch_size"] = 8


class nnUNet_Primus_L_Trainer(AbstractPrimus):

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # this architecture will crash if the patch size is not divisible by 8!
        model = Primus(
            num_input_channels,
            1056,
            (8, 8, 8),
            num_output_channels,
            24,
            16,
            self.configuration_manager.patch_size,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )
        return model


class _Primus_S_96_BS1(nnUNet_Primus_S_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)  # As per repository
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)


class _Primus_B_96_BS1(nnUNet_Primus_B_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)  # As per repository
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)


class _Primus_M_96_BS1(nnUNet_Primus_M_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (96, 96, 96)  # As per repository
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)


class _Primus_L_48_BS1(nnUNet_Primus_L_Trainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plans["configurations"][configuration]["patch_size"] = (48, 48, 48)  # As per repository
        plans["configurations"][configuration]["batch_size"] = 1
        super().__init__(plans, configuration, fold, dataset_json, device)