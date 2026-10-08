from copy import deepcopy
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
from batchgenerators.augmentations.utils import SEG_TIEBREAKS
from einops import rearrange
from torch.nn import functional as F

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis


# Why the seg paths below look the way they do: where scipy beats this implementation on anisotropic
# segmentations (by 2-9x on real data), the reason is the planner's force_separate_z, not the kernel.
# Ideas that were measured and rejected: per-slice sparsity, float16 scores, slab chunking.

SEG_SCALE_FACTOR = 1000


def unique_labels(data: torch.Tensor) -> torch.Tensor:
    """
    torch.unique(data): the distinct values, ascending - but via numpy, and choosing the
    implementation by dtype.

    torch.unique sorts the whole volume, which costs about 0.9 s on a 150 MVoxel int8 label map.
    numpy is 3x quicker at the same thing, and on integer dtypes pd.unique's hash table is quicker
    again (1.3x at int8). For floats pandas loses (4.5x slower at float32 on a label map), and float32
    is what SimpleITKIO hands the preprocessor, so that is the common branch. Next to the per-label
    interpolations this is small either way - at most ~2 % of a call.

    On GPU none of this applies: the sort is parallel there, and going through numpy would mean a
    device transfer.
    """
    if data.device.type != 'cpu':
        return torch.unique(data)
    arr = data.detach().numpy()
    u = np.sort(pd.unique(arr.ravel())) if arr.dtype.kind in 'iub' else np.unique(arr)
    return torch.from_numpy(u)


def _interpolate_into(x: torch.Tensor, new_shape, mode: str, out: torch.Tensor) -> torch.Tensor:
    """
    F.interpolate(x, new_shape, mode=mode, antialias=False) written into `out`. Same aten kernel, so
    the result is bit-identical; F.interpolate has no out= and allocates a fresh output on every call,
    which in the per-label loops below is the largest allocation there is.
    Modes without an out= kernel here (area, bicubic, ...) go through F.interpolate and a copy: correct,
    just without the saved allocation.
    """
    new_shape = [int(i) for i in new_shape]
    nd = len(new_shape)
    if mode == 'trilinear':
        torch.ops.aten.upsample_trilinear3d.out(x, new_shape, False, None, None, None, out=out)
    elif mode == 'bilinear':
        torch.ops.aten.upsample_bilinear2d.out(x, new_shape, False, None, None, out=out)
    elif mode == 'nearest-exact':
        (torch.ops.aten._upsample_nearest_exact3d if nd == 3
         else torch.ops.aten._upsample_nearest_exact2d).out(x, new_shape, *([None] * nd), out=out)
    elif mode == 'nearest':
        (torch.ops.aten.upsample_nearest3d if nd == 3
         else torch.ops.aten.upsample_nearest2d).out(x, new_shape, *([None] * nd), out=out)
    else:
        out.copy_(F.interpolate(x, new_shape, mode=mode, antialias=False))
    return out


def _resample_seg_nearest(data: torch.Tensor, new_shape, mode: str, device) -> torch.Tensor:
    """
    A nearest interpolation of a label map is a gather: it can only copy a value that is already in
    the input, and exactly one label scores at every output voxel, so nothing is ever undecided and no
    tie-break applies. Scoring every label one-hot and taking an argmax - what the loops below do -
    returns the identical answer for n_labels times the work.
    Only the largest label is needed for the output dtype, and a max is a linear reduction where even
    a hash based unique has to build a table.
    """
    result_dtype = torch.int8 if data.max() < 127 else torch.int16
    out = torch.empty((1, data.shape[0], *new_shape), dtype=torch.float32, device=device)
    # the aten nearest kernels have no integer implementation, hence the float32 copy - one copy,
    # not one per label
    _interpolate_into(data[None].float(), new_shape, mode, out)
    return out[0].to(result_dtype)


def _resample_seg_running_argmax(data: torch.Tensor, new_shape, torch_mode: str, seg_tiebreak: str,
                                 unique_values: torch.Tensor, result_dtype, device) -> torch.Tensor:
    """
    Running argmax over the per-label interpolated indicators: memory efficient because only one score
    volume is ever held, rather than the (n_labels, c, *new_shape) buffer the other branch builds.

    Running argmax rather than a `> 0.5` test: where no label reaches 0.5 - an exact tie between two
    labels, or three labels meeting - the threshold writes nothing and the voxel keeps the zero
    `result` was initialized with, which need not even be a label that occurs in the input.

    float16 scores at a scale factor of 1000, deliberately matching the other branch: float16 has a
    spacing of 0.5 at this magnitude, so it collapses near ties into exact ones. A float32 running
    score would resolve some of those by argmax instead and the two branches would disagree on up to
    ~1 % of voxels. Matching it also halves this buffer.

    Every per-label temporary is allocated once here rather than inside the loop. On a full sized
    volume those allocations, not the arithmetic, were the bulk of the cost.
    """
    c = data.shape[0]
    result = torch.zeros((c, *new_shape), dtype=result_dtype, device=device)
    best = torch.zeros((c, *new_shape), dtype=torch.float16, device=device)
    better = torch.empty((c, *new_shape), dtype=torch.bool, device=device)
    scores16 = torch.empty((c, *new_shape), dtype=torch.float16, device=device)
    scores32 = torch.empty((1, c, *new_shape), dtype=torch.float32, device=device)
    onehot = torch.empty((1, *data.shape), dtype=torch.float32, device=device)
    indicator = torch.empty((1, *data.shape), dtype=torch.bool, device=device)

    # `equal` and `scratch` are live only between the float16 cast and the end of the bookkeeping,
    # which is exactly the window in which scores32 holds nothing anyone reads again - the next label
    # overwrites it wholesale. Carving them out of its storage (4 bytes per output voxel, room for
    # both) keeps the peak where it was instead of 2 bytes per output voxel above it.
    # NOTE: this means nothing may read scores32 after the float16 cast. Keep it that way.
    _pool = scores32.view(torch.bool).reshape(-1)
    _n = result.numel()
    equal = _pool[:_n].view(result.shape)
    scratch = _pool[_n:2 * _n].view(result.shape)

    hi = torch.tensor(float(SEG_SCALE_FACTOR), device=device)
    lo = torch.tensor(0., device=device)

    nearest = seg_tiebreak == 'nearest'
    if nearest:
        # An exact tie carries no information, and argmax always resolves it to the first (= smallest)
        # label, which for a normal segmentation is background. Every axis resampled by an even integer
        # factor makes every boundary voxel an exact 50/50 tie, so that systematically erodes foreground
        # and annihilates structures one voxel thick. Nearest neighbour is the decision the separate-z
        # path already makes there: symmetric in the labels, and unbiased in volume.
        # But only if it is one of the tied labels. Where three or more labels meet it can be a label that
        # lost - halving the 2x2x2 block [3,1,1,1,2,2,2,3] scores 1 and 2 at 0.375 each and 3 at 0.25, and
        # the nearest neighbour is a 3. So rather than marking tied voxels, `nn_top` tracks whether the
        # nearest neighbour's label currently has the top score: a label that takes the lead strictly
        # decides that afresh, one that draws level can only add to it. Where it holds at the end the
        # nearest neighbour is the answer (a no-op where its label won outright); elsewhere the argmax
        # winner stands. nn_top is the size of the tied mask it replaces; nn itself adds one result sized
        # volume, since it is needed during the loop now rather than only after it.
        # scores32 holds nothing yet, so the float nearest sample can go through it.
        onehot.copy_(data[None])
        nn = _interpolate_into(onehot, new_shape, 'nearest-exact', out=scores32)[0].to(result_dtype)
        nn_top = torch.zeros((c, *new_shape), dtype=torch.bool, device=device)

    for u in unique_values:
        torch.eq(data[None], u, out=indicator)
        if c > 1:
            # Dim 0 is the channel dim and is never interpolated, so a label absent from a channel
            # scores exactly zero there: it cannot win that channel's argmax and cannot tie either
            # (the tie test below requires > 0). Restricting it to the channel range it occupies is
            # therefore exact, not an approximation, and these are views, so nothing is copied.
            # This is what makes the separate-z path cheap: there the z slices ARE the channel dim,
            # and a label typically lives on a fraction of them.
            present = indicator.any(dim=tuple(range(2, indicator.ndim)))[0].nonzero()
            if len(present) == 0:
                continue
            sl = slice(int(present[0]), int(present[-1]) + 1)
        else:
            sl = slice(None)

        torch.where(indicator[:, sl], hi, lo, out=onehot[:, sl])
        _interpolate_into(onehot[:, sl], new_shape, torch_mode, out=scores32[:, sl])
        tmp = scores16[sl].copy_(scores32[0, sl])
        best_v, better_v, equal_v, scratch_v = best[sl], better[sl], equal[sl], scratch[sl]

        # '>=' lets the later (larger) label take exact ties, '>' leaves them with the earlier one
        (torch.ge if seg_tiebreak == 'highest' else torch.gt)(tmp, best_v, out=better_v)
        if nearest:
            # nn_top = (nn_top and not better) or (this label is the nearest neighbour's and takes the lead
            # or draws level), evaluated against the current leader before `best` is updated. No `> 0`
            # guard is needed on drawing level: the nearest neighbour's own pixel always carries weight, so
            # where this label is the nearest neighbour's its score is positive.
            nn_top_v = nn_top[sl]
            torch.eq(nn[sl], int(u), out=scratch_v)  # an int, so the int8/int16 compare stays integer
            torch.ge(tmp, best_v, out=equal_v)
            equal_v &= scratch_v
            torch.logical_not(better_v, out=scratch_v)
            nn_top_v &= scratch_v
            nn_top_v |= equal_v
        # masked_fill_ and maximum(out=) rather than boolean indexed assignment: `result[better] = u`
        # and `best[better] = tmp[better]` each materialize an index and a gathered copy, which on a
        # full sized output volume costs more than the score buffer this branch exists to avoid.
        result[sl].masked_fill_(better_v, u.item())
        torch.maximum(best_v, tmp, out=best_v)

    if nearest:
        # where, not `result[nn_top] = nn[nn_top]`: nn_top holds nearly everywhere (wherever the nearest
        # neighbour's label won outright, too), and indexing with it materializes an int64 index per voxel
        torch.where(nn_top, nn, result, out=result)
    return result


def resample_torch_simple(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        is_seg: bool = False,
        num_threads: int = 4,
        device: torch.device = torch.device('cpu'),
        memefficient_seg_resampling: bool = True,
        mode='linear',
        *,
        seg_tiebreak: str = 'nearest'
):
    if seg_tiebreak not in SEG_TIEBREAKS:
        # up front, not where a tie first needs settling: that is data dependent, so a typo would pass on
        # some cases and fail on others, part-way through preprocessing a dataset
        raise ValueError(f'unknown seg_tiebreak: {seg_tiebreak}. Must be one of {SEG_TIEBREAKS}')
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

            if is_seg and torch_mode in ('nearest', 'nearest-exact'):
                result = _resample_seg_nearest(data, new_shape, torch_mode, device)
            elif is_seg:
                unique_values = unique_labels(data)
                result_dtype = torch.int8 if max(unique_values) < 127 else torch.int16
                if len(unique_values) == 1:
                    # every output voxel scores the scale factor for the only label there is, so it
                    # wins everywhere and nothing is ever tied
                    result = torch.full((data.shape[0], *new_shape), int(unique_values[0]),
                                        dtype=result_dtype, device=device)
                elif not memefficient_seg_resampling:
                    result = torch.zeros((data.shape[0], *new_shape), dtype=result_dtype, device=device)
                    # This branch scores every label into one (n_labels, c, *new_shape) float16 buffer,
                    # so its peak memory grows with the label count: measured on real data, upsampling
                    # to the full volume, 318 MB for 20 labels and 370 MB for 14 where the branch below
                    # needed 90 MB and 143 MB. It used to be the faster of the two, because it settles most
                    # voxels with a threshold and only argmaxes the uncertain ones, which is why it was the
                    # default. Since the running argmax lost its per-label allocations that is no longer
                    # true: the other branch is faster at every label count measured (1.2-2.8x) and
                    # never needs more memory. The two are bit-identical (see
                    # nnunetv2/tests/test_resampling.py); this one is kept for plans that pin it.

                    # unique_values = torch.unique(data)
                    # result = torch.zeros((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16)
                    # for i, u in enumerate(unique_values):
                    #     result[i] = F.interpolate((data[None] == u).float() * 1000, new_shape, mode='trilinear', antialias=False)[0]
                    # result = unique_values[result.argmax(0)]

                    result_tmp = torch.empty((len(unique_values), data.shape[0], *new_shape), dtype=torch.float16,
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
                        undecided = result_tmp[:, ~done_mask]
                        if seg_tiebreak == 'highest':
                            # argmax reports the first maximum, so reverse the stack to get the last one
                            winner = len(unique_values) - 1 - undecided.flip(0).argmax(0)
                        else:
                            winner = undecided.argmax(0)
                        pick = unique_values[winner].to(result_dtype)
                        if seg_tiebreak == 'nearest':
                            # An exact tie carries no information, and argmax always resolves it to the
                            # first (= smallest) label, which for a normal segmentation is background.
                            # Every axis resampled by an even integer factor makes every boundary voxel
                            # an exact 50/50 tie - in plane just as much as out of plane - so that
                            # systematically erodes foreground and annihilates structures one voxel
                            # thick. Nearest neighbour is the decision the separate-z path already makes
                            # at those voxels: symmetric in the labels, and unbiased in volume.
                            # result_tmp is float16, so a near tie can round into an exact one and take
                            # this branch too. That is harmless - a near tie carries nearly no
                            # information either - but it means this is not exclusively exact ties.
                            # Only if the nearest neighbour is one of the tied labels, though: where three
                            # or more labels meet it can be a label that lost. So it settles exactly the
                            # voxels where its own label has the top score (a no-op where that label won
                            # outright), read off the score stack with a gather.
                            nn = F.interpolate(data[None].float(), new_shape, mode='nearest-exact')[0]
                            nn = nn[~done_mask].to(unique_values.dtype)
                            own = undecided.gather(0, torch.searchsorted(unique_values, nn)[None])[0]
                            use_nn = own == undecided.max(0).values
                            pick[use_nn] = nn[use_nn].to(result_dtype)
                        result[~done_mask] = pick
                else:
                    result = _resample_seg_running_argmax(data, new_shape, torch_mode, seg_tiebreak,
                                                          unique_values, result_dtype, device)
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
        memefficient_seg_resampling: bool = True,
        force_separate_z: Union[bool, None] = None,
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
        mode='linear',
        aniso_axis_mode='nearest-exact',
        *,
        seg_tiebreak: str = 'nearest'
):
    """
    data must be c, x, y, z
    """
    assert data.ndim == 4, "data must be c, x, y, z"
    new_shape = [int(i) for i in new_shape]
    orig_shape = data.shape

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
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
                                     memefficient_seg_resampling=memefficient_seg_resampling, mode=mode,
                                     seg_tiebreak=seg_tiebreak)
        data = rearrange(data, f"(c {axis_letter}) {others[0]} {others[1]} -> c x y z",
                         **{
                             axis_letter: orig_shape[axis + 1],
                             others[0]: tmp_new_shape[0],
                             others[1]: tmp_new_shape[1]
                         }
                         )
        # reshape out of plane. resample_torch_simple recognises the nearest modes and resolves
        # them with a single nearest interpolation instead of scoring every label one-hot.
        data = resample_torch_simple(data, new_shape, is_seg=is_seg, num_threads=num_threads,
                                     device=device,
                                     memefficient_seg_resampling=memefficient_seg_resampling,
                                     mode=aniso_axis_mode, seg_tiebreak=seg_tiebreak)
        if was_numpy:
            data = data.numpy()
        return data
    else:
        return resample_torch_simple(data, new_shape, is_seg=is_seg, num_threads=num_threads, device=device,
                                     memefficient_seg_resampling=memefficient_seg_resampling, mode=mode,
                                     seg_tiebreak=seg_tiebreak)


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
