import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP

'''
WHen using any of these trainers -> Make sure to provide the command line argument `-pretrained_weights` 
to specify the path to the pretrained weights. 
This causes the model to be initialized with the pretrained weights and then the specified layers are fine-tuned.
'''

class nnUNetTrainerFinetune_SegLayers(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Set smaller number of epochs and smaller learning rate
        self.initial_lr = 2e-3
        self.num_epochs = 100

    
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)


            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")        

        
        # Freeze weights in all layers except the segmentation layers
        print("Fine-tuning Segmentation Layers only")
        for name, parameter in self.network.named_parameters():
            if 'seg_layers' in name:
               print(f"parameter '{name}' will not be frozen")
               parameter.requires_grad = True
            else:
               parameter.requires_grad = False
            # parameter.requires_grad = True

        if self._do_i_compile():
            self.print_to_log_file('Compiling network...')
            self.network = torch.compile(self.network)        
    
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank])

        # Print out layers and whether they are frozen or not
        for name, parameter in self.network.named_parameters():
	        print(name, parameter.requires_grad)    


class nnUNetTrainerFinetune_Decoder(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Set smaller number of epochs and smaller learning rate
        self.initial_lr = 2e-3
        self.num_epochs = 100

    
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)


            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")        

        
        # Freeze weights in all layers except the Decoder layers
        print("Fine-tuning Decoder only")
        for name, parameter in self.network.named_parameters():
            if 'decoder' in name:
               print(f"parameter '{name}' will not be frozen")
               parameter.requires_grad = True
            else:
               parameter.requires_grad = False

        if self._do_i_compile():
            self.print_to_log_file('Compiling network...')
            self.network = torch.compile(self.network)        
    
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank])

        # Print out layers and whether they are frozen or not
        for name, parameter in self.network.named_parameters():
	        print(name, parameter.requires_grad)


class nnUNetTrainerFinetune_allWeights(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Set smaller number of epochs and smaller learning rate
        self.initial_lr = 2e-3
        self.num_epochs = 100

    
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                           self.configuration_manager,
                                                           self.num_input_channels,
                                                           enable_deep_supervision=True).to(self.device)


            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")        

        
        # Freeze weights in all layers except the segmentation layers
        print("Fine-tuning all weights")
        for name, parameter in self.network.named_parameters():
            parameter.requires_grad = True

        if self._do_i_compile():
            self.print_to_log_file('Compiling network...')
            self.network = torch.compile(self.network)        
    
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank])

        # Print out layers and whether they are frozen or not
        for name, parameter in self.network.named_parameters():
	        print(name, parameter.requires_grad)                