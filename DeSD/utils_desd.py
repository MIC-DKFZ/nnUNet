# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
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

# TODO: revise this
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


# class Solarization(object):
#     """
#     Apply Solarization to the PIL image.
#     """
#     def __init__(self, p):
#         self.p = p

#     def __call__(self, img):
#         if random.random() < self.p:
#             return ImageOps.solarize(img)
#         else:
#             return img


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
    shutil.copyfile("DeSD/main_DeSD_ssl.py", code_dir/'main_DeSD_ssl.py')
    shutil.copyfile("DeSD/utils_desd.py", code_dir/'utils.py')
    shutil.copyfile("DeSD/data_loader_ssl.py", code_dir/'data_loader_ssl.py')
    shutil.copyfile("DeSD/models/res3d.py", output_dir/'res3d.py')
    shutil.copyfile("DeSD/run_ssl.sh", code_dir/'run_ssl.sh')
    print("files copying finished!")


# class SmoothedValue(object):
#     """Track a series of values and provide access to smoothed values over a
#     window or the global series average.
#     """

#     def __init__(self, window_size=20, fmt=None):
#         if fmt is None:
#             fmt = "{median:.6f} ({global_avg:.6f})"
#         self.deque = deque(maxlen=window_size)
#         self.total = 0.0
#         self.count = 0
#         self.fmt = fmt

#     def update(self, value, n=1):
#         self.deque.append(value)
#         self.count += n
#         self.total += value * n

#     def synchronize_between_processes(self):
#         """
#         Warning: does not synchronize the deque!
#         """
#         if not is_dist_avail_and_initialized():
#             return
#         t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
#         dist.barrier()
#         dist.all_reduce(t)
#         t = t.tolist()
#         self.count = int(t[0])
#         self.total = t[1]

#     @property
#     def median(self):
#         d = torch.tensor(list(self.deque))
#         return d.median().item()

#     @property
#     def avg(self):
#         d = torch.tensor(list(self.deque), dtype=torch.float32)
#         return d.mean().item()

#     @property
#     def global_avg(self):
#         return self.total / self.count

#     @property
#     def max(self):
#         return max(self.deque)

#     @property
#     def value(self):
#         return self.deque[-1]

#     def __str__(self):
#         return self.fmt.format(
#             median=self.median,
#             avg=self.avg,
#             global_avg=self.global_avg,
#             max=self.max,
#             value=self.value)


# def reduce_dict(input_dict, average=True):
#     """
#     Args:
#         input_dict (dict): all the values will be reduced
#         average (bool): whether to do average or sum
#     Reduce the values in the dictionary from all processes so that all processes
#     have the averaged results. Returns a dict with the same fields as
#     input_dict, after reduction.
#     """
#     world_size = get_world_size()
#     if world_size < 2:
#         return input_dict
#     with torch.no_grad():
#         names = []
#         values = []
#         # sort the keys so that they are consistent across processes
#         for k in sorted(input_dict.keys()):
#             names.append(k)
#             values.append(input_dict[k])
#         values = torch.stack(values, dim=0)
#         dist.all_reduce(values)
#         if average:
#             values /= world_size
#         reduced_dict = {k: v for k, v in zip(names, values)}
#     return reduced_dict


# class MetricLogger(object):
#     def __init__(self, delimiter="\t"):
#         self.meters = defaultdict(SmoothedValue)
#         self.delimiter = delimiter

#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             assert isinstance(v, (float, int))
#             self.meters[k].update(v)

#     def __getattr__(self, attr):
#         if attr in self.meters:
#             return self.meters[attr]
#         if attr in self.__dict__:
#             return self.__dict__[attr]
#         raise AttributeError("'{}' object has no attribute '{}'".format(
#             type(self).__name__, attr))

#     def __str__(self):
#         loss_str = []
#         for name, meter in self.meters.items():
#             loss_str.append(
#                 "{}: {}".format(name, str(meter))
#             )
#         return self.delimiter.join(loss_str)

#     def synchronize_between_processes(self):
#         for meter in self.meters.values():
#             meter.synchronize_between_processes()

#     def add_meter(self, name, meter):
#         self.meters[name] = meter

#     def log_every(self, iterable, print_freq, header=None):
#         i = 0
#         if not header:
#             header = ''
#         start_time = time.time()
#         end = time.time()
#         iter_time = SmoothedValue(fmt='{avg:.6f}')
#         data_time = SmoothedValue(fmt='{avg:.6f}')
#         space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
#         if torch.cuda.is_available():
#             log_msg = self.delimiter.join([
#                 header,
#                 '[{0' + space_fmt + '}/{1}]',
#                 'eta: {eta}',
#                 '{meters}',
#                 'time: {time}',
#                 'data: {data}',
#                 'max mem: {memory:.0f}'
#             ])
#         else:
#             log_msg = self.delimiter.join([
#                 header,
#                 '[{0' + space_fmt + '}/{1}]',
#                 'eta: {eta}',
#                 '{meters}',
#                 'time: {time}',
#                 'data: {data}'
#             ])
#         MB = 1024.0 * 1024.0
#         for obj in iterable:
#             data_time.update(time.time() - end)
#             yield obj
#             iter_time.update(time.time() - end)
#             if i % print_freq == 0 or i == len(iterable) - 1:
#                 eta_seconds = iter_time.global_avg * (len(iterable) - i)
#                 eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#                 if torch.cuda.is_available():
#                     print(log_msg.format(
#                         i, len(iterable), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time),
#                         memory=torch.cuda.max_memory_allocated() / MB))
#                 else:
#                     print(log_msg.format(
#                         i, len(iterable), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time)))
#             i += 1
#             end = time.time()
#         total_time = time.time() - start_time
#         total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#         print('{} Total time: {} ({:.6f} s / it)'.format(
#             header, total_time_str, total_time / len(iterable)))

#     def log_every_v2(self, iterable_list, print_freq, header=None):
#         i = 0
#         iterable1, iterable2 = iterable_list
#         if not header:
#             header = ''
#         start_time = time.time()
#         end = time.time()
#         iter_time = SmoothedValue(fmt='{avg:.6f}')
#         data_time = SmoothedValue(fmt='{avg:.6f}')
#         space_fmt = ':' + str(len(str(len(iterable1)))) + 'd'
#         if torch.cuda.is_available():
#             log_msg = self.delimiter.join([
#                 header,
#                 '[{0' + space_fmt + '}/{1}]',
#                 'eta: {eta}',
#                 '{meters}',
#                 'time: {time}',
#                 'data: {data}',
#                 'max mem: {memory:.0f}'
#             ])
#         else:
#             log_msg = self.delimiter.join([
#                 header,
#                 '[{0' + space_fmt + '}/{1}]',
#                 'eta: {eta}',
#                 '{meters}',
#                 'time: {time}',
#                 'data: {data}'
#             ])
#         MB = 1024.0 * 1024.0
#         for obj in zip(iterable1, iterable2):
#             data_time.update(time.time() - end)
#             yield obj
#             iter_time.update(time.time() - end)
#             if i % print_freq == 0 or i == len(iterable1) - 1:
#                 eta_seconds = iter_time.global_avg * (len(iterable1) - i)
#                 eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#                 if torch.cuda.is_available():
#                     print(log_msg.format(
#                         i, len(iterable1), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time),
#                         memory=torch.cuda.max_memory_allocated() / MB))
#                 else:
#                     print(log_msg.format(
#                         i, len(iterable1), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time)))
#             i += 1
#             end = time.time()
#         total_time = time.time() - start_time
#         total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#         print('{} Total time: {} ({:.6f} s / it)'.format(
#             header, total_time_str, total_time / len(iterable1)))


# def get_sha():
#     cwd = os.path.dirname(os.path.abspath(__file__))

#     def _run(command):
#         return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
#     sha = 'N/A'
#     diff = "clean"
#     branch = 'N/A'
#     try:
#         sha = _run(['git', 'rev-parse', 'HEAD'])
#         subprocess.check_output(['git', 'diff'], cwd=cwd)
#         diff = _run(['git', 'diff-index', 'HEAD'])
#         diff = "has uncommited changes" if diff else "clean"
#         branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
#     except Exception:
#         pass
#     message = f"sha: {sha}, status: {diff}, branch: {branch}"
#     return message


# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True


# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()


# def is_main_process():
#     return get_rank() == 0


# def save_on_master(*args, **kwargs):
#     if is_main_process():
#         torch.save(*args, **kwargs)


# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__
#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print


# def init_distributed_mode(args):
#     usgpu = args.use_single_gpu
#     print(usgpu)
#     if (not usgpu) and ('RANK' in os.environ) and ('WORLD_SIZE' in os.environ):
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     # launched with submitit on a slurm cluster
#     elif (not usgpu) and ('SLURM_PROCID' in os.environ):
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = args.rank % torch.cuda.device_count()
#     # launched naively with `python main_DeSD_ssl.py`
#     # we manually add MASTER_ADDR and MASTER_PORT to env variables
#     elif torch.cuda.is_available():
#         print('Will run the code on one GPU.')
#         args.rank, args.gpu, args.world_size = 0, 0, 1
#         os.environ['MASTER_ADDR'] = '127.0.0.1'
#         os.environ['MASTER_PORT'] = '29500'
#     else:
#         print('Does not support training without GPU.')
#         sys.exit(1)

#     dist.init_process_group(
#         backend="nccl",
#         init_method=args.dist_url,
#         world_size=args.world_size,
#         rank=args.rank,
#     )

#     torch.cuda.set_device(args.gpu)
#     print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
#     dist.barrier()
#     # fix the printing behaviour in "master"
#     setup_for_distributed(args.rank == 0)


def adapt_chckpt(src_path: Path, dst_path: Path):
    src = torch.load(src_path)
    state_dict = {'network_weights': {}}

    for k, v in src['teacher'].items():
        if 'backbone.net' in k:
            k = str(k).replace('backbone.net', 'encoder')
        state_dict['network_weights'][k] = v

    state_dict['network_weights'] = OrderedDict(state_dict['network_weights'])
    torch.save(state_dict, dst_path)


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.reshape(1, -1).expand_as(pred))
#     return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


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


# def has_batchnorms(model):
#     bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
#     for name, module in model.named_modules():
#         if isinstance(module, bn_types):
#             return True
#     return False


# class Weight_adaptive(nn.Module):
#     def __init__(self, embed_dim):
#         super(Weight_adaptive, self).__init__()
#         self.embedding = torch.nn.Embedding(4, embed_dim)
#         self.controller = nn.Linear(embed_dim, 1)

#     def forward(self, x):
#         embedding = self.embedding(x)
#         weights = torch.softmax(self.controller(embedding).squeeze(-1), dim=0)
#         return weights, embedding

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