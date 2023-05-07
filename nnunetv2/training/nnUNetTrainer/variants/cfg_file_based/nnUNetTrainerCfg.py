import os
import torch
import yaml
import numpy as np
from pathlib import Path
from time import time
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.logging.nnunet_logger import nnUNetLoggerTB
from torch.nn.parallel import DistributedDataParallel as DDP

class nnUNetTrainerCfg(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        dataset_name = plans['dataset_name']
        nnunet_cfg_path = Path(nnUNet_preprocessed)/ dataset_name / 'nnunet_cfg.yml'
        with open(nnunet_cfg_path, 'r') as yfile:
            self.nnunet_cfg = yaml.safe_load(yfile)

        self.num_epochs = self.nnunet_cfg['num_epochs']
        self.ssl_pretrained = self.nnunet_cfg['ssl_pretrained']
        print(f'Using pretrained enconder: {self.ssl_pretrained}')
        self.unfreeze_lr = self.nnunet_cfg['unfreeze_lr'] if self.ssl_pretrained else None
        self.unfreeze_epoch = self.nnunet_cfg['unfreeze_epoch'] if self.ssl_pretrained else None
        self.tb_log_file = self.log_file.replace('.txt', '_tb.log')
        self.logger = nnUNetLoggerTB(self.tb_log_file)
        

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)
            if (self.unfreeze_lr is not None) or (self.unfreeze_epoch is not None):
                # initialy freeze encoder weights
                for param in self.network.encoder.parameters():
                    param.requires_grad = False

            model_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            total_params = sum([np.prod(p.size()) for p in self.network.parameters()])
            print(f'When initializing - The model has {(params/total_params)*100}% of trainable parameters ')

            # compile network for free speedup
            if ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't')):
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        lr = self.optimizer.param_groups[0]['lr']
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(lr, decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', lr, self.current_epoch)

        # If we reach the criteria, unfreeze encoder
        if self.unfreeze_lr is not None:
            if lr <= self.unfreeze_lr:
                for param in self.network.encoder.parameters():
                    param.requires_grad = True
                self.unfreeze_lr = None
                self.print_to_log_file(
                    f"\nUnfreezing encoder at epoch: {self.current_epoch} and lr: {lr}")
                model_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
                params = sum([np.prod(p.size()) for p in model_parameters])
                total_params = sum([np.prod(p.size()) for p in self.network.parameters()])
                print(f'When unfreezing - The model has {(params/total_params)*100}% of trainable parameters ')
        elif self.unfreeze_epoch is not None:
            if self.current_epoch == self.unfreeze_epoch:
                for param in self.network.encoder.parameters():
                    param.requires_grad = True
                self.unfreeze_epoch = None
                self.print_to_log_file(
                    f"\nUnfreezing encoder at epoch: {self.current_epoch} and lr: {lr}")
                model_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
                params = sum([np.prod(p.size()) for p in model_parameters])
                total_params = sum([np.prod(p.size()) for p in self.network.parameters()])
                print(f'When unfreezing - The model has {(params/total_params)*100}% of trainable parameters ')


    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                            self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
        
        if (current_epoch + 1) in self.nnunet_cfg['save_at_epochs']:
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch + 1}.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1