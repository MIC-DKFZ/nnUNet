from copy import deepcopy
from typing import Union, Tuple, List

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis


def resample_torch_simple(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        is_seg: bool = False,
        num_threads: int = 4,
        device: torch.device = torch.device('cpu'),
        memefficient_seg_resampling: bool = False,
        mode='linear'
):
    if mode == 'linear':
        if data.ndim == 4:
            torch_mode = 'trilinear'
        elif data.ndim == 3:
            torch_mode = 'bilinear'
        else:
            raise RuntimeError
    else:
        torch_mode = mode

    if isinstance(new_shape, np.ndarray):
        new_shape = [int(i) for i in new_shape]

    if all([i == j for i, j in zip(new_shape, data.shape[1:])]):
        return data
    else:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads)
        new_shape = tuple(new_shape)
        with torch.no_grad():

            input_was_numpy = isinstance(data, np.ndarray)
            if input_was_numpy:
                data = torch.from_numpy(data).to(device)
            else:
                orig_device = deepcopy(data.device)
                data = data.to(device)

            if is_seg:
                unique_values = torch.unique(data)
                result_dtype = torch.int8 if max(unique_values) < 127 else torch.int16
                result = torch.zeros((data.shape[0], *new_shape), dtype=result_dtype, device=device)
                if not memefficient_seg_resampling:
                    # believe it or not, the implementation below is 3x as fast (at least on Liver CT and on CPU)
                    # Why? Because argmax is slow. The implementation below immediately sets most locations and only lets the
                    # uncertain ones be determined by argmax

                    # unique_values = torch.unique(data)
                    # result = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16)
                    # for i, u in enumerate(unique_values):
                    #     result[i] = F.interpolate((data[None] == u).float() * 1000, new_shape, mode='trilinear', antialias=False)[0]
                    # result = unique_values[result.argmax(0)]

                    result_tmp = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16,
                                             device=device)
                    scale_factor = 1000
                    done_mask = torch.zeros_like(result, dtype=torch.bool, device=device)
                    for i, u in enumerate(unique_values):
                        result_tmp[i] = \
                            F.interpolate((data[None] == u).float() * scale_factor, new_shape, mode=torch_mode,
                                          antialias=False)[0]
                        mask = result_tmp[i] > (0.7 * scale_factor)
                        result[mask] = u.item()
                        done_mask |= mask
                    if not torch.all(done_mask):
                        # print('resolving argmax', torch.sum(~done_mask), "voxels to go")
                        result[~done_mask] = unique_values[result_tmp[:, ~done_mask].argmax(0)].to(result_dtype)
                else:
                    for i, u in enumerate(unique_values):
                        if u == 0:
                            pass
                        result[F.interpolate((data[None] == u).float(), new_shape, mode=torch_mode, antialias=False)[
                                   0] > 0.5] = u
            else:
                result = F.interpolate(data[None].float(), new_shape, mode=torch_mode, antialias=False)[0]
            if input_was_numpy:
                result = result.cpu().numpy()
            else:
                result = result.to(orig_device)
        torch.set_num_threads(n_threads)
        return result


def resample_torch_fornnunet(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        is_seg: bool = False,
        num_threads: int = 4,
        device: torch.device = torch.device('cpu'),
        memefficient_seg_resampling: bool = False,
        force_separate_z: Union[bool, None] = None,
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
        mode='linear',
        aniso_axis_mode='nearest-exact'
):
    """
    data must be c, x, y, z
    """
    assert data.ndim == 4, "data must be c, x, y, z"
    new_shape = [int(i) for i in new_shape]
    orig_shape = data.shape

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)
    # print('shape', data.shape, 'current_spacing', current_spacing, 'new_spacing', new_spacing, 'do_separate_z', do_separate_z, 'axis', axis)

    if do_separate_z:
        was_numpy = isinstance(data, np.ndarray)
        if was_numpy:
            data = torch.from_numpy(data)

        assert len(axis) == 1
        axis = axis[0]
        tmp = "xyz"
        axis_letter = tmp[axis]
        others_int = [i for i in range(3) if i != axis]
        others = [tmp[i] for i in others_int]

        # reshape by overloading c channel
        data = rearrange(data, f"c x y z -> (c {axis_letter}) {others[0]} {others[1]}")

        # reshape in-plane
        tmp_new_shape = [new_shape[i] for i in others_int]
        data = resample_torch_simple(data, tmp_new_shape, is_seg=is_seg, num_threads=num_threads, device=device,
                                     memefficient_seg_resampling=memefficient_seg_resampling, mode=mode)
        data = rearrange(data, f"(c {axis_letter}) {others[0]} {others[1]} -> c x y z",
                         **{
                             axis_letter: orig_shape[axis + 1],
                             others[0]: tmp_new_shape[0],
                             others[1]: tmp_new_shape[1]
                         }
                         )
        # reshape out of plane w/ nearest
        data = resample_torch_simple(data, new_shape, is_seg=is_seg, num_threads=num_threads, device=device,
                                     memefficient_seg_resampling=memefficient_seg_resampling, mode=aniso_axis_mode)
        if was_numpy:
            data = data.numpy()
        return data
    else:
        return resample_torch_simple(data, new_shape, is_seg, num_threads, device, memefficient_seg_resampling)


if __name__ == '__main__':
    torch.set_num_threads(16)
    img_file = '/media/isensee/raw_data/nnUNet_raw/Dataset027_ACDC/imagesTr/patient041_frame01_0000.nii.gz'
    seg_file = '/media/isensee/raw_data/nnUNet_raw/Dataset027_ACDC/labelsTr/patient041_frame01.nii.gz'
    io = SimpleITKIO()
    data, pkl = io.read_images((img_file, ))
    seg, pkl = io.read_seg(seg_file)

    target_shape = (15, 256, 312)
    spacing = pkl['spacing']

    use = data
    is_seg = False

    ret_nosep = resample_torch_fornnunet(use, target_shape, spacing, spacing, is_seg)
    ret_sep = resample_torch_fornnunet(use, target_shape, spacing, spacing, is_seg, force_separate_z=False)

