import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!

    network can be either a plain model or DDP. We need to account for that in the parameter names
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']
    is_ddp = isinstance(network, DDP)

    model_dict = network.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if is_ddp:
            key_pretrained = key[7:]
        else:
            key_pretrained = key
        if '.seg_layers.' not in key:
            assert key_pretrained in pretrained_dict, \
                f"Key {key_pretrained} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key_pretrained].shape, \
                f"The shape of the parameters of key {key_pretrained} is not the same. Pretrained model: " \
                f"{pretrained_dict[key_pretrained].shape}; your network: {model_dict[key]}"

    # fun fact this does allow loading from parameters that do not cover the entire network. For example pretrained
    # encoders
    # I didnt even know that you could put if statements into dict comprehensions. Damn am I good.
    pretrained_dict = {'module.' + k if is_ddp else k: v
                       for k, v in pretrained_dict.items()
                       if (('module.' + k if is_ddp else k) in model_dict) and '.seg_layers.' not in k}
    # yet another line of death right here...

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, _ in pretrained_dict.items():
            print(key[7:] if is_ddp else key)
        print("################### Done ###################")
    network.load_state_dict(model_dict)


