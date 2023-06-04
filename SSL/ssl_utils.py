import argparse
import os
import sys
import time
import math
import random
import datetime
import logging
import subprocess
import shutil
import torch
import numpy as np
import torch.distributed as dist
from torch import nn
from PIL import ImageFilter, ImageOps
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
import SimpleITK as sitk
from typing import Union, List, Dict


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}')
    else:
        print("There is no reference weights available for this model => We use random weights.")


def clip_gradients(model, clip):
    norms = []
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def load_encoder_from_checkpoint(encoder_chkpt: str, student: nn.Module):
    encoder_chkpt = Path(encoder_chkpt)
    assert encoder_chkpt.exists(), 'Enocder chkpt path does not exist'

    print(f'Found checkpoint at {encoder_chkpt}')
    checkpoint = torch.load(encoder_chkpt, map_location="cpu")
    desired_weights = {}
    for key, val in checkpoint['student'].items():
        if 'backbone' not in key:
            continue
        desired_weights[key.replace('backbone.', '')] = val
    msg = student.load_state_dict(desired_weights, strict=False)
    return student


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded {} from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value
    schedule += 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def copy_code(output_dir: Path):
    code_dir = output_dir / 'code'
    code_dir.mkdir(exist_ok=True, parents=True)
    shutil.copyfile("SSL/main_SSL_ssl.py", code_dir/'main_SSL_ssl.py')
    shutil.copyfile("SSL/ssl_utils.py", code_dir/'utils.py')
    shutil.copyfile("SSL/data_loader_ssl.py", code_dir/'data_loader_ssl.py')
    shutil.copyfile("SSL/models/res3d.py", output_dir/'res3d.py')
    shutil.copyfile("SSL/run_ssl.sh", code_dir/'run_ssl.sh')
    print("files copying finished!")


def adapt_chckpt(src_path: Path, dst_path: Path):
    src = torch.load(src_path)
    state_dict = {'network_weights': {}}

    for k, v in src['teacher'].items():
        if 'backbone.net' in k:
            k = str(k).replace('backbone.net', 'encoder')
        state_dict['network_weights'][k] = v

    state_dict['network_weights'] = OrderedDict(state_dict['network_weights'])
    torch.save(state_dict, dst_path)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        logging.warning(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """type: (Tensor, float, float, float, float) -> Tensor"""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm),
                                                one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def get_params_groups_encoder(model, lr):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        # we do not regularize biases nor Norm parameters
        if 'backbone' in name:
            param.requires_grad = True
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
    return [{'params': regularized, 'enc': True},
            {'params': not_regularized, 'weight_decay': 0., 'enc': True}]


def copy_metadata_from_reference(
    img: sitk.Image, reference: sitk.Image
) -> sitk.Image:
    """Copies metadata from one image to the other.
    Args:
        img (sitk.Image): sitk image to be modified
        reference (sitk.Image): sitk image to use as reference
    Returns:
        sitk.Image: modified image.
    """
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    img.SetSpacing(reference.GetSpacing())
    for key in reference.GetMetaDataKeys():
        img.SetMetaData(key, reference.GetMetaData(key))
    return img


def save_img_from_array_using_referece(
    volume: Union[np.ndarray, List[np.ndarray]], reference: sitk.Image, filepath: Path,
    ref_3d_for_4d: bool = False
) -> None:
    """Stores the volume in nifty format using the spatial parameters coming
        from a reference image
    Args:
        volume (np.ndarray | List[np.ndarray]): Volume to store in Nifty format.
            If a list of 3d volumes is passed, then a single 4d nifty is stored.
        reference (sitk.Image): Reference image to get the spatial parameters from.
        filepath (Path): Where to save the volume.
        ref_3d_for_4d (bool, optional): Whether to use 3d metadata and store a 4d image.
            Defaults to False
    """
    meta_ready = False

    # determine if the image is a 3d or 4d image
    if (type(volume) == list) or (len(volume.shape) > 3):
        # if the image is 4d determine the right reference and store
        if type(volume[0]) == sitk.Image:
            vol_list = [vol for vol in volume]
        else:
            vol_list = [sitk.GetImageFromArray(vol) for vol in volume]
            if ref_3d_for_4d:
                vol_list = [copy_metadata_from_reference(vol, reference) for vol in vol_list]
                meta_ready = True
        joiner = sitk.JoinSeriesImageFilter()
        img = joiner.Execute(*vol_list)
    else:
        if isinstance(volume, np.ndarray):
            img = sitk.GetImageFromArray(volume)
        elif not isinstance(volume, sitk.Image):
            raise Exception(
                'Error in save_img_from_array_using_referece, '
                'passed image is neither an array or a sitk.Image'
            )

    # correct the metadata using reference image
    if not meta_ready:
        copy_metadata_from_reference(img, reference)
    # write
    sitk.WriteImage(img, str(filepath))