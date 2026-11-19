"""
Memory-efficient (streaming) inference for nnU-Net.

The default nnU-Net inference path materializes the full (num_classes, *internal_shape) logit volume,
then resamples it as a whole to (num_classes, *target_shape), then runs argmax/region-thresholding.
For datasets with many classes (e.g. 120) and large images (e.g. 512x512x1024) the resampled probability
volume alone can require >100 GB of RAM.

nnUNetPredictorStreaming fuses sliding-window prediction, resampling and segmentation export into a single
streaming pass along one spatial axis ("slab axis"). It never holds the full (num_classes, *target_shape)
volume. The key facts that make this exact:

  1. trilinear/bilinear resampling (torch F.interpolate) is independent per channel, so we can resample
     channel-wise / slab-wise.
  2. logits -> segmentation is a per-voxel cross-channel reduction (argmax for softmax, ordered threshold
     for sigmoid/regions) that streams with O(one output slab) memory.
  3. the sliding-window patch loop already visits patches with the first spatial axis as the outermost loop,
     so once all patches with start <= z (along the slab axis) have been accumulated, every source slice
     below the next patch start is final (fully accumulated). We reorder the loop so the slab axis is the
     outermost one.

Resampling is separable: bilinear in-plane (per source slice) + linear along the slab axis, which is torch
trilinear (align_corners=False). Where the plans resample probabilities with nnU-Net's separate-z path
(resample_torch_fornnunet or resample_data_or_seg_to_shape with force_separate_z None/True) and that path fires
for a case, streaming follows it: the low-resolution axis takes the nearest source slice instead of a linear
blend, chosen by the same determine_do_sep_z_and_axis call the default export makes. Anything streaming cannot
reproduce (e.g. cubic probability resampling) falls back to plain trilinear, and force_trilinear=True
(--force_trilinear) always does. A notice is printed at runtime either way.

Supports 2d and 3d configurations and fold ensembling (logits averaged per slab). No probability export.
Softmax and region-based (sigmoid) models: argmax over the logits, or for regions the ordered threshold of
LabelManager.convert_probabilities_to_segmentation on the logits (sigmoid(x) > 0.5 <=> x > 0, regions painted in
regions_class_order, the last one wins); the sigmoid itself is never computed.

On CUDA with Triton, accumulation and the per-plane resample + argmax / region threshold run as fused kernels
(streaming_kernels.py) that reproduce the torch path bit for bit; the torch path is the fallback everywhere else.

Feature ring (3d, default on where it applies). The network ends in a per-voxel linear layer (decoder.seg_layers[-1],
a 1x1x1 convolution with bias), and everything between that layer and the argmax / region threshold is linear: mirror
averaging, Gaussian weighting, the sum over tiles and folds, the division by the weight sum, linear resampling. So
    sum_t g_t (W f_t + b) / sum_t g_t = W (sum_t g_t f_t / sum_t g_t) + b
and the ring can hold the layer's input f (the decoder's last features, 32 channels for the standard UNets) instead of
its output (one channel per class). The layer is detached from the network before it is compiled and applied per
finished source slice, in float32. A K-fold ensemble keeps one block of features per fold and applies the folds'
layers side by side, which is exact as well; the ring then has K x features channels, so it is used only while that is
fewer than the class count (else the logit ring). Same segmentation up to floating-point rounding (the sums
run in a different order). feature_ring=False / --logit_ring keeps the logits.

Standalone inference (the predict_* entry points and the CLI) runs with cudnn.benchmark off and compiles the network
by policy (resolve_compile): 'auto' compiles for batches of more than AUTO_COMPILE_MAX_CASES cases (where torch.compile
is available: CUDA + Triton) and not for single images, where compiling costs more than it saves. A predictor on CUDA
enables expandable allocator segments unless an allocator configuration is set or CUDA is already initialized. A
trainer's final validation (manual_initialization -> _predict_and_export_streaming) keeps the trainer's settings.

Batch prediction (predict_cases_streaming and the predict_from_* wrappers) overlaps loading + preprocessing in
background workers, the GPU pass in the main process, and export in background workers; the worker pools are owned by
the predictor and reused across calls, and the input / output handling is pluggable (load_fn / export_fn). A single
case runs in-process, without any pool. See streaming_pipeline.py.

Per case, the preprocessed input volume is moved to the device once and each patch is cut from it (+4 bytes per input
voxel of VRAM, no host-to-device copy per patch); the weights of a single model stay loaded across cases, 2d fold
ensembles keep one network per fold across cases, 3d fold ensembles swap weights per slab from pinned host copies
(fold_weights_on_device / --fold_weights_on_device: from copies on the device, faster, + one model of VRAM per fold).
The CUDA allocator cache is emptied once per call of a public entry point (also on error), not per case.
"""
import contextlib
import copy
import inspect
import io
import itertools
import os
import time
import weakref
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from threading import Thread
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, save_json
from torch.nn import functional as F

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.streaming_pipeline import WORKER_DIED_MESSAGE, CaseFailure, PreprocessingIterator, \
    collect_results, load_and_preprocess, load_npy_case, new_pool, shutdown_pools, timed_export, write_segmentation, \
    _init_preprocessing_worker
from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis, \
    resample_data_or_seg_to_shape
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.checkpoints import find_checkpoint_file, load_checkpoint_file
from nnunetv2.utilities.helpers import empty_cache, dummy_context, is_optimized_module, unwrap_network, \
    set_default_inductor_cache_dir
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

# compile policy 'auto' (resolve_compile): compile for batches of more than this many cases, never for fewer
AUTO_COMPILE_MAX_CASES = 4

# printed by nnUNetv2_predict_streaming and written to the training log by streaming validation
STREAMING_CITATION = ("Isensee, F., Wasserthal, J., Shiyam Sundar, L. K., Rokuss, M., Ulrich Harsy, C., "
                      "Langenberg, M., Happold, S. M., & Maier-Hein, K. H. (2026). Segment Everything, Materialize "
                      "Nothing: Exact Streaming Inference for Many-Class 3D Segmentation. arXiv:XXXX.XXXXX.")


def _ring_segments(start: int, stop: int, capacity: int) -> List[Tuple[int, int, int, int]]:
    """
    Map the absolute index range [start, stop) onto a circular buffer of size ``capacity``.

    Returns a list of (offset_in_range, ring_lo, ring_hi, length) tuples. ``offset_in_range`` is the
    position within [start, stop) (i.e. within the source patch/band), the ring slice is [ring_lo, ring_hi).
    A range that wraps the buffer boundary yields two segments.
    """
    assert stop > start
    segments = []
    pos = start
    while pos < stop:
        slot = pos % capacity
        run = min(stop - pos, capacity - slot)
        segments.append((pos - start, slot, slot + run, run))
        pos += run
    return segments


def separate_z_rule(resampling_fn) -> Tuple[bool, Union[str, None]]:
    """
    Can streaming reproduce this resampling_fn_probabilities (a partial, as the ConfigurationManager hands it
    out), and how does its separate-z path pick a source slice along the low-resolution axis?

    Returns (reproducible, rule). rule is 'nearest-exact' / 'nearest' (torch's kernels) or 'scipy'
    (resample_data_or_seg, order_z=0); None means the separate-z path is linear along that axis too, i.e. plain
    trilinear. Not reproducible (False, None): streaming resamples plain trilinear.
    """
    fn = getattr(resampling_fn, 'func', resampling_fn)
    try:
        params = {k: v.default for k, v in inspect.signature(resampling_fn).parameters.items()}
    except (TypeError, ValueError):
        return False, None
    if fn is resample_torch_fornnunet:
        if params.get('mode') != 'linear':
            return False, None
        aniso = params.get('aniso_axis_mode')
        if aniso in ('nearest-exact', 'nearest'):
            return True, aniso
        return (True, None) if aniso == 'linear' else (False, None)
    if fn is resample_data_or_seg_to_shape:
        # in-plane: skimage resize order 1 (mode edge, no anti-aliasing) = bilinear with half-pixel centres;
        # along z: order_z 0 = nearest (take), order_z 1 = linear (map_coordinates) -> plain trilinear
        if params.get('order') != 1 or params.get('order_z') not in (0, 1):
            return False, None
        return True, ('scipy' if params.get('order_z') == 0 else None)
    return False, None


def nearest_source_index(source_len: int, target_len: int, rule: str) -> torch.Tensor:
    """
    For every target index along the low-resolution axis, the source index the separate-z path copies.
    The torch rules are read off the kernel itself (float32 arithmetic and all), so they agree with
    resample_torch_fornnunet by construction; 'scipy' is the index computation of resample_data_or_seg.
    """
    if rule == 'scipy':
        scale = float(source_len) / target_len
        coords = scale * (np.arange(target_len) + 0.5) - 0.5
        return torch.from_numpy(np.clip(np.floor(coords + 0.5).astype(np.int64), 0, source_len - 1))
    src = torch.arange(source_len, dtype=torch.float32)[None, None]
    return F.interpolate(src, size=target_len, mode=rule)[0, 0].round().long()


def compile_available(device: torch.device) -> Tuple[bool, str]:
    """torch.compile needs CUDA and Triton here; nnU-Net's trainer draws the same line (no MPS, no CPU, no Windows)."""
    if device.type != 'cuda':
        return False, f'device is {device.type}'
    if os.name == 'nt':
        return False, 'Windows has no Triton wheels'
    try:
        import triton  # noqa: F401
    except ImportError:
        return False, 'Triton is not installed'
    if torch.cuda.get_device_capability(device) < (7, 0):   # Triton's minimum (torch.utils._triton.has_triton)
        return False, f'GPU compute capability {torch.cuda.get_device_capability(device)} < 7.0'
    return True, ''


def resolve_compile(requested: Union[bool, str, None], device: torch.device, num_cases: Optional[int] = None) -> bool:
    """
    Whether standalone streaming inference compiles the network for a call that predicts num_cases cases (None:
    unknown). Precedence: an explicit request (True / False, CLI --compile on/off) over the nnUNet_compile environment
    variable over the 'auto' policy (requested 'auto' or None): compile for batches of more than AUTO_COMPILE_MAX_CASES
    cases wherever torch.compile is available, never for single images or an unknown number of cases. Compiling costs
    seconds on the first case (~4 s with a cold cache, ~2.8 s warm, 3d nets on an RTX 4090) and saves 7-17 % per case
    afterwards, so it pays only over several cases; the cache persists (set_default_inductor_cache_dir).
    An explicit request that cannot be honoured raises; the environment variable degrades to a notice, as in the
    trainer.
    """
    available, why = compile_available(device)
    if requested is not None and requested != 'auto':
        if requested and not available:
            raise RuntimeError(f'torch.compile was requested but is not available: {why}')
        return bool(requested)
    env = os.environ.get('nnUNet_compile')
    if env is not None:
        on = env.lower() in ('true', '1', 't')
        if on and not available:
            print(f'INFO: nnUNet_compile is set but torch.compile is not available ({why}); running without it')
            return False
        return on
    return available and num_cases is not None and num_cases > AUTO_COMPILE_MAX_CASES


def maybe_enable_expandable_segments() -> bool:
    """
    Expandable CUDA allocator segments let the allocator reuse freed memory instead of caching blocks it cannot fit the
    next request into: the reserved peak then follows the allocated one (otherwise up to ~2 GB higher at 192^3 x 126
    classes). Takes effect only before CUDA is initialized, and never overrides an allocator configuration the user set
    (PYTORCH_CUDA_ALLOC_CONF takes precedence over PYTORCH_ALLOC_CONF in torch, so check both). Returns whether it was
    set by this call.
    """
    if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ or 'PYTORCH_ALLOC_CONF' in os.environ or torch.cuda.is_initialized():
        return False
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return True


def find_linear_head(network: torch.nn.Module) -> Tuple[Union[str, None], str]:
    """
    (qualified name, '') of the network's final per-voxel linear layer if the network is one whose forward we know
    ends in it - dynamic_network_architectures' UNets: forward = decoder(encoder(x)), and their decoders return
    seg_layers[-1](last features) when deep supervision is off - else (None, why not). The layer must be a 1x1(x1)
    convolution (stride 1, no padding, groups 1).
    """
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet, ResidualUNet
        from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
        from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
    except ImportError:
        return None, 'dynamic_network_architectures is not importable'
    network = unwrap_network(network)
    if type(network).forward not in (PlainConvUNet.forward, ResidualEncoderUNet.forward, ResidualUNet.forward):
        return None, f'{type(network).__name__} is not one of the UNets whose output layer is known'
    decoder = network.decoder
    if not isinstance(decoder, (UNetDecoder, UNetResDecoder)) or \
            type(decoder).forward not in (UNetDecoder.forward, UNetResDecoder.forward):
        return None, f'unknown decoder {type(decoder).__name__}'
    if decoder.deep_supervision:
        return None, 'deep supervision is enabled'
    head = decoder.seg_layers[-1]
    if not isinstance(head, (torch.nn.Conv2d, torch.nn.Conv3d)) or head.groups != 1 or \
            any(k != 1 for k in head.kernel_size) or any(s != 1 for s in head.stride) or \
            any(p != 0 for p in head.padding):
        return None, f'the output layer is not a 1x1 convolution ({head})'
    return f'decoder.seg_layers.{len(decoder.seg_layers) - 1}', ''


class nnUNetPredictorStreaming(nnUNetPredictor):
    """
    A derivative of nnUNetPredictor that performs sliding-window prediction, resampling to the original
    resolution and segmentation export in a single streaming pass that keeps only a thin band of the
    (num_classes, *internal_shape) volume in GPU memory at any time.

    force_trilinear=True resamples plain trilinear even where the plans' separate-z path would fire (module
    docstring).
    compile: 'auto' (default; None means the same) compiles for batches of more than AUTO_COMPILE_MAX_CASES cases and
    not for single images; True / False force it on / off (resolve_compile). set_compile changes it later.
    feature_ring: accumulate the decoder's last features instead of the class logits wherever that is exact and the
    smaller ring (module docstring); False keeps the logits.
    num_processes_preprocessing / num_processes_export: default sizes of the persistent worker pools of batch
    prediction (streaming_pipeline.py); close() shuts them down (also: use the predictor as a context manager).
    quiet: print nothing from the main process (worker processes print nothing unless verbose_preprocessing).
    expandable_segments: enable expandable CUDA allocator segments (maybe_enable_expandable_segments) on CUDA.
    fold_weights_on_device: fold ensembles keep every fold's weights on the device (+ one model of VRAM per fold)
    instead of in pinned host memory: faster weight swaps (_place_fold_weights). No effect for a single model or 2d.

    A predictor is meant to be kept: initialize it once, then call the predict_* methods as often as needed. Batches
    overlap preprocessing, GPU and export (predict_cases_streaming); single cases run in-process.
    """
    # ring capacity = patch extent along the slab axis + this. Slices are emitted right after each slab is flushed,
    # so only the next target plane's lower neighbour has to survive the following flush. (Private; tests raise it
    # to check that the ring size does not change the output.)
    _ring_margin = 1
    # fused Triton kernels where available and kernels_selftest passes (private; tests switch them off for the reference)
    _use_triton_kernels = True
    # CPU threads for torch resampling when a single case is preprocessed in this process (None: every core). Batch
    # workers keep the plans' setting: there the workers run side by side.
    single_case_preprocessing_threads: Optional[int] = None

    def __init__(self, *args, force_trilinear: bool = False, compile: Union[bool, str, None] = 'auto',
                 feature_ring: bool = True, num_processes_preprocessing: int = 3, num_processes_export: int = 3,
                 quiet: bool = False, expandable_segments: bool = True, fold_weights_on_device: bool = False,
                 **kwargs):
        self.quiet = quiet
        with self._quiet():
            super().__init__(*args, **kwargs)
        if self.device.type == 'cuda' and expandable_segments:
            maybe_enable_expandable_segments()
        self.force_trilinear = force_trilinear
        self.set_compile(compile)
        self.feature_ring = feature_ring
        self.num_processes_preprocessing = num_processes_preprocessing
        self.num_processes_export = num_processes_export
        self.fold_weights_on_device = fold_weights_on_device
        # 2d fold ensembles: (identity key, one network per fold), see _fold_networks_2d
        self._fold_nets_2d = None
        # set by detach_head while the network's output layer is detached (feature ring), else None
        self._detached_head = None
        self.feature_ring_status = 'not set up'
        # background weight loading (initialize_from_folder_base_trainer(load_weights_in_background=True)), else None
        self._weights_loader = None
        # per-case stage timings (seconds) of the last predict_* call, see predict_from_data_iterator_streaming
        self.last_case_timings = []
        # persistent worker pools {name: (key, ProcessPoolExecutor)}; the finalizer holds the dict, not the predictor
        self._pools = {}
        self._pools_finalizer = weakref.finalize(self, shutdown_pools, self._pools)

    # ------------------------------------------------------------------ lifecycle -------------------------------------
    def close(self):
        """Shut down the persistent worker pools. The predictor stays usable: pools are recreated when needed."""
        shutdown_pools(self._pools)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _quiet(self):
        """Silences the main process's stdout while quiet is set (also nnUNetPredictor's own messages)."""
        return contextlib.redirect_stdout(io.StringIO()) if getattr(self, 'quiet', False) else contextlib.nullcontext()

    def set_compile(self, compile: Union[bool, str, None]):
        """'auto' / None, True or False (see the class docstring); takes effect at the next prediction call."""
        assert compile in (True, False, 'auto', None), f"compile must be True, False or 'auto', got {compile!r}"
        self.compile = 'auto' if compile is None else compile

    def _pool(self, name: str, num_processes: int, initializer=None, initargs: tuple = (), key: tuple = ()):
        """The persistent pool `name`, (re)created if it does not exist yet or its key (size, model) changed."""
        key = (num_processes, *key)
        current = self._pools.get(name)
        if current is not None and current[0] == key:
            return current[1]
        if current is not None:
            self._pools.pop(name)[1].shutdown(wait=True, cancel_futures=True)
        pool = new_pool(num_processes, initializer, initargs)
        self._pools[name] = (key, pool)
        return pool

    def _drop_pool(self, name: str):
        """A pool whose worker died is broken for good; forget it so the next call starts a fresh one."""
        current = self._pools.pop(name, None)
        if current is not None:
            current[1].shutdown(wait=False, cancel_futures=True)

    def _preprocessing_pool(self, num_processes: int):
        # the workers hold the plans, so the pool is tied to the model it was created for
        model = (id(self.plans_manager), id(self.configuration_manager), id(self.dataset_json), self.verbose_preprocessing)
        return self._pool('preprocessing', num_processes, _init_preprocessing_worker,
                          (self.plans_manager.plans, self.configuration_manager.configuration, self.dataset_json,
                           self.verbose_preprocessing), model)

    def _export_pool(self, num_processes: int):
        return self._pool('export', num_processes)

    # ------------------------------------------------------------------ weights ---------------------------------------
    def _ensure_weights(self):
        """Wait for weights loading in the background (initialize_from_folder_base_trainer) and install them."""
        loader = self._weights_loader
        if loader is None:
            return
        loader['thread'].join()
        self._weights_loader = None
        if loader['error'] is not None:
            raise RuntimeError('Loading the model weights failed') from loader['error']
        self.list_of_parameters = loader['parameters']

    def _prepare_standalone_inference(self, num_cases: Optional[int] = 1):
        """
        Settings for standalone inference, applied by the public predict_* entry points, not by a trainer's final
        validation. cudnn.benchmark off: its autotuning costs ~0.6 s per process plus a one-off VRAM peak (~1 GB) for
        a forward pass only ~1.6 % faster, amortised after ~7 large cases. Feature ring: the output layer is detached
        (detach_head) before compiling, so the compiled graph ends at the decoder's last features. torch.compile per
        resolve_compile for a call of num_cases cases; a network compiled at initialization (nnUNet_compile=True) is
        unwrapped again if compile=False was requested, and recompiled if the output layer was detached from it here.
        Under the 'auto' policy a network that is compiled already (an earlier batch) stays compiled: the cost is paid.
        """
        self._ensure_weights()
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
        want = resolve_compile(self.compile, self.device, num_cases)
        compiled = is_optimized_module(self.network)
        if compiled and not want and self.compile == 'auto' and 'nnUNet_compile' not in os.environ:
            want = True
        self._place_fold_weights()                          # before detach_head: the stripped dicts share them
        detached_now = self.detach_head()
        if want and (not compiled or detached_now):
            print('Using torch.compile')
            set_default_inductor_cache_dir()
            self.network = torch.compile(unwrap_network(self.network))
        elif compiled and not want:
            print('torch.compile switched off again (compile=False / --compile off overrides nnUNet_compile)')
            self.network = self.network._orig_mod

    # ------------------------------------------------------------------ feature ring ----------------------------------
    def _feature_ring_applicable(self) -> Tuple[Union[str, None], str]:
        """(name of the output layer to detach, '') if the feature ring applies to this predictor, else (None, why)."""
        if not self.feature_ring:
            return None, 'switched off (feature_ring=False / --logit_ring)'
        if len(self.configuration_manager.patch_size) != 3:
            return None, '2d configuration (no ring buffer to shrink)'
        name, why = find_linear_head(self.network)
        if name is None:
            return None, why
        head = unwrap_network(self.network).get_submodule(name)
        num_folds = len(self.list_of_parameters)
        if num_folds * head.in_channels >= head.out_channels:
            return None, (f'{num_folds} fold(s) x {head.in_channels} features >= {head.out_channels} output channels: '
                          f'the logit ring is smaller')
        return name, ''

    def detach_head(self) -> bool:
        """
        Feature ring: replace the network's output layer by an identity, so that the network returns the decoder's last
        features, and keep the layer's weights per fold for _predict_and_export_streaming. Fold weights come from
        list_of_parameters (their output-layer entries move into the head, so loading a fold stays a strict
        load_state_dict), or from the network itself where list_of_parameters holds None (a trainer's live network).
        Unwraps DDP / torch.compile; whoever compiled the network must recompile it (the compiled graph would retrace
        anyway: dynamo guards on the module structure). Returns True if the layer was detached by this call, False if
        it already was or the feature ring does not apply (feature_ring_status says why).
        """
        self._ensure_weights()
        net = unwrap_network(self.network)
        if self._detached_head is not None:
            if self._detached_head['network'] is net:
                return False
            self._detached_head = None                       # a different network was set since: start over
        name, why = self._feature_ring_applicable()
        if name is None:
            self.feature_ring_status = f'off: {why}'
            return False
        parent_name, idx = name.rsplit('.', 1)
        parent = net.get_submodule(parent_name)
        module = parent[int(idx)]
        heads, params = [], []
        for p in self.list_of_parameters:
            if p is None:
                w, b = module.weight.detach(), module.bias
                params.append(None)
            else:
                w, b = p[name + '.weight'], p.get(name + '.bias')
                params.append({k: v for k, v in p.items() if not k.startswith(name + '.')})
            heads.append((w.detach().flatten(1).float().clone(),
                          b.detach().float().clone() if b is not None else torch.zeros(w.shape[0])))
        parent[int(idx)] = torch.nn.Identity()
        self._detached_head = {'network': net, 'parent': parent, 'idx': int(idx), 'module': module,
                               'heads': heads, 'list_of_parameters': self.list_of_parameters, 'stacked': None}
        self.list_of_parameters = params
        self.feature_ring_status = (f'on: the ring holds {len(heads)} x {module.in_channels} decoder features instead '
                                    f'of {module.out_channels} logits')
        return True

    def restore_head(self):
        """Undo detach_head: the output layer goes back into the network (with the weights of the fold the network holds
        now), list_of_parameters gets its full state dicts back. No-op if nothing is detached."""
        d = self._detached_head
        if d is None:
            return
        module = d['module']
        loaded = self._loaded_parameters          # (module, state dict) the network holds, see _load_fold
        fold = next((i for i, p in enumerate(self.list_of_parameters) if loaded is not None and p is loaded[1]
                     and loaded[0]() is d['network']), None)
        if fold is not None and d['list_of_parameters'][fold] is not None:
            W, b = d['heads'][fold]
            with torch.no_grad():
                module.weight.copy_(W.view_as(module.weight))
                if module.bias is not None:
                    module.bias.copy_(b)
        d['parent'][d['idx']] = module
        self.list_of_parameters = d['list_of_parameters']
        self._detached_head = None
        self.feature_ring_status = 'off: output layer restored'

    def _stacked_head(self, device: torch.device):
        """(W (C, K*F) fp32, b (C,) fp32) on device: the K folds' output layers side by side, mean bias. With each
        fold's features in its own block of the ring and the weight sum accumulated once per fold (n = K * sum_t g_t),
        (W f + b n) / n = mean_k (W_k f_k / sum_t g_t + b_k), the logit ring's ensemble mean."""
        d = self._detached_head
        if d['stacked'] is None or d['stacked'][0].device != device:
            W = torch.cat([w for w, _ in d['heads']], 1).to(device)
            b = torch.stack([b for _, b in d['heads']], 0).mean(0).to(device)
            d['stacked'] = (W, b)
        return d['stacked']

    def _logit_methods_unavailable(self):
        if self._detached_head is not None:
            raise RuntimeError('The network\'s output layer is detached for the streaming feature ring, so it returns '
                               'features, not logits. Call restore_head() first, or construct the predictor with '
                               'feature_ring=False.')

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        self._logit_methods_unavailable()
        return super().predict_logits_from_preprocessed_data(data)

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        self._logit_methods_unavailable()
        return super().predict_sliding_window_return_logits(input_image)

    def separate_z_for_case(self, properties: dict) -> Union[Tuple[int, str], None]:
        """
        (low-resolution axis, index rule) if the configured probability resampling takes its separate-z path for
        this case and streaming reproduces it; None means plain trilinear. The decision is the default export's
        own: the same determine_do_sep_z_and_axis call on the same spacings (export_prediction.
        convert_predicted_logits_to_segmentation_with_correct_shape).
        """
        if self.force_trilinear:
            return None
        cm = self.configuration_manager
        fn = cm.resampling_fn_probabilities
        reproducible, rule = separate_z_rule(fn)
        if not reproducible or rule is None:
            return None
        params = inspect.signature(fn).parameters
        spacing_transposed = [properties['spacing'][i] for i in self.plans_manager.transpose_forward]
        current_spacing = cm.spacing if len(cm.spacing) == len(properties['shape_after_cropping_and_before_resampling']) \
            else [spacing_transposed[0], *cm.spacing]
        # axes the plans do not resample (spacing None) keep the case's own spacing, as in the default export
        current_spacing = [s if c is None else c for c, s in zip(current_spacing, spacing_transposed)]
        do_separate_z, axis = determine_do_sep_z_and_axis(params['force_separate_z'].default, current_spacing,
                                                          spacing_transposed,
                                                          params['separate_z_anisotropy_threshold'].default)
        return (int(axis), rule) if do_separate_z else None

    # ------------------------------------------------------------------ public API ------------------------------------
    @torch.inference_mode()
    def predict_single_npy_array_streaming(self,
                                           input_image: np.ndarray,
                                           image_properties: dict,
                                           output_file_truncated: Union[str, None] = None,
                                           slab_axis: Union[int, None] = None,
                                           ring_dtype: torch.dtype = torch.half):
        """
        Preprocess a single image (in the nnU-Net expected axis ordering, see nnUNetPredictor.predict_single_npy_array)
        and run the streaming inference pipeline. If output_file_truncated is given the segmentation is written
        (file ending appended automatically), otherwise the segmentation (numpy, original orientation) is returned.
        """
        return self.predict_single_case_streaming((input_image, image_properties), load_npy_case, output_file_truncated,
                                                  slab_axis=slab_axis, ring_dtype=ring_dtype)

    @torch.inference_mode()
    def predict_single_case_streaming(self, case: Any, load_fn: Optional[Callable] = None, output: Any = None,
                                      export_fn: Optional[Callable] = None, export_kwargs: Optional[dict] = None,
                                      slab_axis: Union[int, None] = None, ring_dtype: torch.dtype = torch.half,
                                      on_error: str = 'raise', progress_callback: Optional[Callable] = None):
        """
        One case, entirely in this process - no worker pools to start: load_fn(case) (None: case is a list of image
        files, read with the plans' reader) and nnU-Net's preprocessing, the streaming GPU pass, then
        export_fn(segmentation, properties, output, **export_kwargs) (streaming_pipeline.py). Without export_fn the
        segmentation is returned if output is None, else written with nnU-Net's writer to output + file ending.
        Returns export_fn's result (a CaseFailure under on_error='skip' if a stage failed). Stage timings go to
        last_case_timings, progress_callback(event, index, total) as in predict_from_data_iterator_streaming. Loading
        and preprocessing run in a background thread while this one loads the weights onto the GPU and warms it up
        (warm_up); timings 'gpu_ready' and 'wait_for_preprocessing' show which of the two was waited for.
        quiet silences the predictor's own messages, not those of load_fn and export_fn (as in batch prediction, where
        they run in the worker processes).
        """
        assert on_error in ('raise', 'skip'), f"on_error must be 'raise' or 'skip', got {on_error!r}"
        notify = progress_callback or (lambda *args: None)
        timings = {}
        self.last_case_timings = [timings]
        stage = 'load'
        try:
            # The case is loaded and preprocessed in a background thread (CPU; reading, numpy and torch release the
            # GIL, torch resampling gets every core) while this thread gets the GPU ready (warm_up: it has to be the
            # thread that runs the network). The GPU pass starts when both are done.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='nnunet_preprocessing') as background:
                loading = background.submit(self._load_and_preprocess_single_case, load_fn, case)
                stage = 'predict'
                t = time.perf_counter()
                with self._quiet():
                    self._prepare_standalone_inference(num_cases=1)
                    self.warm_up(ring_dtype)
                timings['gpu_ready'] = time.perf_counter() - t
                stage = 'load'
                t = time.perf_counter()
                data, properties, timings['load_and_preprocess'] = loading.result()
                timings['wait_for_preprocessing'] = time.perf_counter() - t
            notify('preprocessed', 0, 1)
            stage = 'predict'
            with self._quiet():
                t = time.perf_counter()
                segmentation = self._predict_and_export_streaming(torch.from_numpy(data), properties,
                                                                  slab_axis=slab_axis, ring_dtype=ring_dtype)
                timings['predict'] = time.perf_counter() - t
            notify('predicted', 0, 1)
            stage = 'export'
            t = time.perf_counter()
            if export_fn is None and output is None:
                result = segmentation
            elif export_fn is None:
                result = write_segmentation(segmentation, properties, output, **self._default_export_kwargs())
            else:
                result = export_fn(segmentation, properties, output, **(export_kwargs or {}))
            timings['export'] = time.perf_counter() - t
            notify('exported', 0, 1)
            return result
        except Exception as e:
            if on_error == 'raise':
                raise
            notify('failed', 0, 1)
            return CaseFailure.from_exception(0, stage, e)
        finally:
            empty_cache(self.device)                   # once per entry-point call (as _cache_emptied_after)

    def _load_and_preprocess_single_case(self, load_fn: Optional[Callable], case: Any):
        """load_and_preprocess for the in-process single case (run in a background thread), with every core for torch:
        its own operations and torch resampling. The CLIs set one torch thread on CUDA for the GPU loop; torch's OpenMP
        thread count is per thread once set, so raising it here leaves that thread alone. Returns (data, properties,
        seconds)."""
        t = time.perf_counter()
        threads = self.single_case_preprocessing_threads or os.cpu_count()
        previous = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
            data, properties = load_and_preprocess(load_fn, case, self.plans_manager, self.configuration_manager,
                                                   self.dataset_json, preprocessor, resampling_threads=threads)
        finally:
            torch.set_num_threads(previous)
        return data, properties, time.perf_counter() - t

    @torch.inference_mode()
    def warm_up(self, ring_dtype: torch.dtype = torch.half):
        """
        Get the GPU ready for the first case without one, in the calling thread: CUDA context, the network and the first
        fold's weights on the device, one forward pass on a zero patch through the same call as the streaming pass (CUDA
        kernels load; cuDNN finds its execution plans, which it caches per thread - so this has to run in the thread
        that runs the network later; a compiled network compiles here), the Gaussian tile weighting (cached), the
        feature ring's output layer (cuBLAS) and the streaming kernels' self-test. The forward's memory stays in the allocator's cache for the first case. The
        single-case path runs this while the case is preprocessed in a background thread, the batch path while its
        workers start. Costs one patch of GPU time. CUDA and 3d configurations only (elsewhere a no-op); call after
        _prepare_standalone_inference.
        """
        if self.device.type != 'cuda' or len(self.configuration_manager.patch_size) != 3:
            return
        self._gaussian()                                     # cached for the streaming pass
        self.network = self.network.to(self.device).eval()
        self._load_fold(0)
        num_inputs = determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)
        x = torch.zeros((1, num_inputs, *self.configuration_manager.patch_size), device=self.device)
        with torch.autocast(self.device.type, enabled=True):
            out = self._internal_maybe_mirror_and_predict(x)[0]
        fr = self._detached_head
        if fr is not None and fr['network'] is unwrap_network(self.network):
            head_W, _ = self._stacked_head(self.device)
            f = torch.zeros((head_W.shape[1], out.shape[-2] * out.shape[-1]), device=self.device)
            with torch.autocast(self.device.type, enabled=False):
                torch.mm(head_W, f)
            del f
        self._streaming_kernels(ring_dtype)                  # their self-test, once per process
        del x, out
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict_cases_streaming(self, cases: Sequence[Any], load_fn: Optional[Callable] = None,
                                outputs: Optional[Sequence[Any]] = None, export_fn: Optional[Callable] = None,
                                export_kwargs: Optional[dict] = None, num_processes_preprocessing: Optional[int] = None,
                                num_processes_export: Optional[int] = None, slab_axis: Union[int, None] = None,
                                ring_dtype: torch.dtype = torch.half, on_error: str = 'raise',
                                progress_callback: Optional[Callable] = None, num_parts: int = 1,
                                part_id: int = 0) -> List[Any]:
        """
        Batch prediction with pluggable input and output handling (streaming_pipeline.py): persistent preprocessing
        workers run load_fn(case) + nnU-Net's preprocessing a few cases ahead, the GPU pass runs here, and persistent
        export workers run export_fn(segmentation, properties, output, **export_kwargs). load_fn None: each case is a
        list of image files read with the plans' reader. export_fn None: cases with an output (truncated file name) are
        written with nnU-Net's writer, cases without one return their segmentation. A single case runs in-process
        (predict_single_case_streaming). Pool sizes default to the predictor's.
        on_error: 'raise' (default) or 'skip' (failed cases become CaseFailure results, the batch carries on).
        progress_callback(event, index, total): see predict_from_data_iterator_streaming. num_parts / part_id: this
        call predicts cases[part_id::num_parts] (one process per GPU). Returns one result per predicted case, in order;
        their stage timings are in last_case_timings.
        """
        assert 0 <= part_id < num_parts, 'part_id must be in 0 .. num_parts - 1'
        assert outputs is None or len(outputs) == len(cases), 'one output per case'
        cases = list(cases)[part_id::num_parts]
        outputs = None if outputs is None else list(outputs)[part_id::num_parts]
        if len(cases) == 0:
            self.last_case_timings = []
            return []
        if len(cases) == 1:
            return [self.predict_single_case_streaming(cases[0], load_fn, None if outputs is None else outputs[0],
                                                       export_fn, export_kwargs, slab_axis, ring_dtype, on_error,
                                                       progress_callback)]
        data_iterator = self.get_data_iterator_streaming(cases, load_fn, outputs, num_processes_preprocessing,
                                                         on_error)
        return self.predict_from_data_iterator_streaming(data_iterator, export_fn, export_kwargs, num_processes_export,
                                                         slab_axis=slab_axis, ring_dtype=ring_dtype,
                                                         on_error=on_error, progress_callback=progress_callback)

    def get_data_iterator_streaming(self, cases: Sequence[Any], load_fn: Optional[Callable] = None,
                                    outputs: Optional[Sequence[Any]] = None, num_processes: Optional[int] = None,
                                    on_error: str = 'raise') -> PreprocessingIterator:
        """Preprocessed items of cases (see predict_cases_streaming), from the persistent preprocessing pool."""
        num_processes = max(1, num_processes or self.num_processes_preprocessing)
        return PreprocessingIterator(lambda: self._preprocessing_pool(num_processes), cases, load_fn, outputs,
                                     max_in_flight=2 * num_processes,
                                     on_broken=lambda: self._drop_pool('preprocessing'), on_error=on_error)

    @torch.inference_mode()
    def predict_from_data_iterator_streaming(self, data_iterator, export_fn: Optional[Callable] = None,
                                             export_kwargs: Optional[dict] = None,
                                             num_processes_export: Optional[int] = None,
                                             num_cases: Optional[int] = None, slab_axis: Union[int, None] = None,
                                             ring_dtype: torch.dtype = torch.half, on_error: str = 'raise',
                                             progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        The batch engine. data_iterator yields {'data': preprocessed (c, *internal shape) array or tensor (or the path
        of a .npy file, deleted after loading), 'data_properties': dict, 'ofile': output or None}, as nnU-Net's
        preprocessing iterators do (get_data_iterator_streaming's items also carry 'index', 'timings', or 'error').
        Each item gets the streaming GPU pass here and is handed to export_fn in the persistent export pool (export_fn
        None: see predict_cases_streaming); at most num_processes_export + 2 exports wait at any time, so finished
        segmentations cannot pile up in RAM. num_cases (default: len(data_iterator) where it has one) feeds the compile
        policy. Returns one result per item, in order; with on_error='skip' a failed case is a CaseFailure.
        Per-item stage timings (seconds: 'load_and_preprocess', 'predict', 'export') go to last_case_timings.
        progress_callback(event, index, total), event in 'preprocessed', 'predicted', 'exported', 'failed', is called
        in this process - for 'exported' of pooled exports from a background thread of it; total may be None.
        """
        assert on_error in ('raise', 'skip'), f"on_error must be 'raise' or 'skip', got {on_error!r}"
        with self._quiet(), self._cache_emptied_after():
            if num_cases is None and hasattr(data_iterator, '__len__'):
                num_cases = len(data_iterator)
            notify = progress_callback or (lambda *args: None)
            num_export = max(1, num_processes_export or self.num_processes_export)
            default_export = export_fn is None
            if default_export:
                export_fn, export_kwargs = write_segmentation, self._default_export_kwargs()
            export_kwargs = export_kwargs or {}
            on_broken = lambda: self._drop_pool('export')
            results, timings, indices, prepared = [], [], [], False
            if hasattr(data_iterator, 'start'):
                # its workers start and preprocess the first cases while the weights load and the GPU warms up here
                # (and a compiled network compiles); other iterators may preprocess lazily on the first next(), so
                # for them this waits until their first case arrives
                data_iterator.start()
                try:
                    self._prepare_standalone_inference(num_cases)
                    self.warm_up(ring_dtype)
                except BaseException:
                    if hasattr(data_iterator, 'cancel'):
                        data_iterator.cancel()
                    raise
                prepared = True
            for position, item in enumerate(data_iterator):
                index = item.get('index', position)
                indices.append(index)
                timings.append(dict(item.get('timings', {})))
                if 'error' in item:                 # loading / preprocessing failed under on_error='skip'
                    results.append(item['error'])
                    notify('failed', index, num_cases)
                    continue
                notify('preprocessed', index, num_cases)
                if not prepared:   # only now: the workers spawned and preprocessed meanwhile, weights may have loaded
                    self._prepare_standalone_inference(num_cases)
                    prepared = True
                data = item['data']
                if isinstance(data, str):
                    delfile = data
                    data = np.load(data)
                    os.remove(delfile)
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                data = data.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                properties, ofile = item['data_properties'], item['ofile']
                print(f'\nPredicting {os.path.basename(ofile)}:' if isinstance(ofile, str) else
                      f'\nPredicting image of shape {tuple(data.shape)}:')
                self._wait_for_exports(results, num_export, on_broken)
                t = time.perf_counter()
                try:
                    segmentation = self._predict_and_export_streaming(data, properties, slab_axis=slab_axis,
                                                                      ring_dtype=ring_dtype)
                except Exception as e:
                    if on_error == 'raise':
                        raise
                    del data
                    empty_cache(self.device)         # a failed case (e.g. out of memory) frees its memory for the next
                    results.append(CaseFailure.from_exception(index, 'predict', e))
                    notify('failed', index, num_cases)
                    continue
                timings[-1]['predict'] = time.perf_counter() - t
                notify('predicted', index, num_cases)
                if default_export and ofile is None:
                    results.append(segmentation)
                    notify('exported', index, num_cases)
                else:
                    try:
                        future = self._export_pool(num_export).submit(timed_export, export_fn, segmentation,
                                                                      properties, ofile, export_kwargs)
                    except BrokenProcessPool as e:
                        on_broken()
                        raise RuntimeError(WORKER_DIED_MESSAGE) from e
                    if progress_callback is not None:
                        future.add_done_callback(
                            lambda f, i=index: notify('exported' if f.exception() is None else 'failed', i,
                                                      num_cases))
                    results.append(future)
                del data, segmentation
            print('\nGPU prediction completed. Waiting for remaining exports to finish...')
            results = collect_results(results, on_broken, timings, on_error, indices)
            self.last_case_timings = timings
            print('Export complete.')
        # compute_gaussian's lru_cache is kept (maxsize=2: at most two small maps): later calls reuse it
        return results

    @torch.inference_mode()
    def predict_from_list_of_npy_arrays_streaming(self, image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                                  properties_or_list_of_properties: Union[dict, List[dict]],
                                                  output_file_truncated: Union[str, List[str], None] = None,
                                                  num_processes_preprocessing: Optional[int] = None,
                                                  num_processes_export: Optional[int] = None,
                                                  slab_axis: Union[int, None] = None,
                                                  ring_dtype: torch.dtype = torch.half) -> List[Any]:
        """
        Streaming counterpart of nnUNetPredictor.predict_from_list_of_npy_arrays (no cascades, no probabilities): images
        (c, x, y, z) in the axis convention of the model's reader, properties with at least 'spacing'. Without output
        files the segmentations are returned (in the images' geometry), else written. Batches as predict_cases_streaming.
        """
        images = image_or_list_of_images if isinstance(image_or_list_of_images, list) else [image_or_list_of_images]
        properties = properties_or_list_of_properties if isinstance(properties_or_list_of_properties, list) else \
            [properties_or_list_of_properties]
        outputs = [output_file_truncated] if isinstance(output_file_truncated, str) else output_file_truncated
        return self.predict_cases_streaming(list(zip(images, properties)), load_npy_case, outputs,
                                            num_processes_preprocessing=num_processes_preprocessing,
                                            num_processes_export=num_processes_export, slab_axis=slab_axis,
                                            ring_dtype=ring_dtype)

    def _default_export_kwargs(self) -> dict:
        return {'reader_writer_class': self.plans_manager.image_reader_writer_class,
                'file_ending': self.dataset_json['file_ending']}

    @staticmethod
    def _wait_for_exports(results: list, num_export: int, on_broken: Callable[[], None]):
        """Backpressure, as nnUNetPredictor's export pool: proceed once fewer than num_export + 2 exports are pending."""
        while True:
            pending = [r for r in results if hasattr(r, 'done') and not r.done()]
            if len(pending) < num_export + 2:
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            if any(isinstance(f.exception(), BrokenProcessPool) for f in done):
                on_broken()
                raise RuntimeError(WORKER_DIED_MESSAGE)

    def initialize_from_folder_base_trainer(self, model_training_output_dir: str,
                                            use_folds: Union[Tuple[Union[int, str], ...], None],
                                            checkpoint_name: str = 'checkpoint_final.safetensors',
                                            load_weights_in_background: bool = False):
        """
        Like nnUNetPredictor.initialize_from_trained_model_folder, but builds the network with the base
        nnUNetTrainer instead of looking up the (possibly custom / external) trainer class named in the
        checkpoint. This is safe for inference whenever the trainer variant does not change the architecture
        (data-augmentation / epoch / mirroring variants don't) - the architecture is fully specified by the plans.
        Mirror axes are still taken from the checkpoint, so TTA stays correct. load_state_dict would fail loudly
        if the architecture did not match.

        load_weights_in_background: read only the checkpoints' metadata now and the network weights in a background
        thread, so the caller can load and preprocess its first case meanwhile (the predict_* methods wait for the
        weights only once preprocessing is done). Applies to .safetensors checkpoints; a .pth checkpoint has to be
        unpickled whole to read its metadata, so .pth weights are loaded right away.
        """
        # nnUNetTrainer.build_network_architecture is exactly this call; importing the trainer module would pull in
        # the training stack (losses -> torch._dynamo, ~0.4 s per process) for nothing
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        with self._quiet():
            if self._weights_loader is not None:          # a previous model still loading: let it finish, drop it
                self._weights_loader['thread'].join()
                self._weights_loader = None
            if use_folds is None:
                use_folds = nnUNetPredictorStreaming.auto_detect_available_folds(model_training_output_dir,
                                                                                 checkpoint_name)
            if isinstance(use_folds, str):
                use_folds = [use_folds]
            files = [join(model_training_output_dir, f'fold_{int(f) if f != "all" else f}', checkpoint_name)
                     for f in use_folds]

            dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
            plans_manager = PlansManager(load_json(join(model_training_output_dir, 'plans.json')))

            resolved = [find_checkpoint_file(f) for f in files]
            background = load_weights_in_background and all(r is not None and r.endswith('.safetensors')
                                                            for r in resolved)
            meta_keys = ('trainer_name', 'init_args', 'inference_allowed_mirroring_axes')

            def load_weights(fnames):
                return [load_checkpoint_file(f, map_location=torch.device('cpu'), keys=('network_weights',))[
                            'network_weights'] for f in fnames]

            first = load_checkpoint_file(files[0], map_location=torch.device('cpu'),
                                         keys=meta_keys if background else ('network_weights', *meta_keys))
            trainer_name = first['trainer_name']
            configuration_name = first['init_args']['configuration']
            inference_allowed_mirroring_axes = first.get('inference_allowed_mirroring_axes', None)
            parameters = None if background else [first['network_weights'], *load_weights(files[1:])]
            del first

            configuration_manager = plans_manager.get_configuration(configuration_name)
            num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
            num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
            network = get_network_from_plans(configuration_manager.network_arch_class_name,
                                             configuration_manager.network_arch_init_kwargs,
                                             configuration_manager.network_arch_init_kwargs_req_import,
                                             num_input_channels, num_output_channels, allow_init=True,
                                             deep_supervision=False)
            self.manual_initialization(network, plans_manager, configuration_manager, parameters, dataset_json,
                                       trainer_name, inference_allowed_mirroring_axes)
            if background:
                loader = {'parameters': None, 'error': None}

                def run():
                    try:
                        loader['parameters'] = load_weights(files)
                    except BaseException as e:      # re-raised in the main thread by _ensure_weights
                        loader['error'] = e
                loader['thread'] = Thread(target=run, daemon=True)
                loader['thread'].start()
                self._weights_loader = loader

    def print_streaming_deviation_notice(self, slab_axis: Union[int, None] = None):
        """
        Print how this run predicts, so the user can see that it reproduces nnUNetv2_predict, and where it does not
        (resampling that streaming cannot reproduce, or force_trilinear), what differs. The most important item is how
        the predicted logits are resampled: the plans' resampling where streaming reproduces it, plain torch trilinear
        otherwise.
        """
        from nnunetv2.configuration import ANISO_THRESHOLD
        cfg = self.configuration_manager.configuration
        fn = cfg.get('resampling_fn_probabilities')
        kw = cfg.get('resampling_fn_probabilities_kwargs')
        spacing = [s for s in self.configuration_manager.spacing if s is not None]   # None: axis not resampled
        aniso_ratio = (max(spacing) / min(spacing)) if (spacing and min(spacing) > 0) else 1.0

        reproducible, _ = separate_z_rule(self.configuration_manager.resampling_fn_probabilities)
        follows_plans = reproducible and not self.force_trilinear
        lines = ["", "#" * 92, "# nnUNetv2_predict_streaming - memory-efficient streaming inference"]
        if self.label_manager.has_regions:
            lines += ["# Prediction, resampling and the region thresholds run in one pass over thin slabs of the",
                      "# image, so the full probability volume is never held in memory."]
        else:
            lines += ["# Prediction, resampling and argmax run in one pass over thin slabs of the image, so the",
                      "# full probability volume is never held in memory."]
        if follows_plans:
            lines += ["# Segmentations match nnUNetv2_predict up to floating-point rounding at label boundaries.",
                      "#",
                      f"#  * Resampling as configured in the plans: {fn} with kwargs {kw}",
                      "#    (separate-z decided per case, as in nnUNetv2_predict)."]
        else:
            why = 'force_trilinear is set' if self.force_trilinear else 'streaming cannot reproduce the plans\' resampling'
            lines += ["# Same as nnUNetv2_predict except for how the predicted logits are resampled:",
                      "#",
                      f"#  ! Resampling ({why}): the plans configure {fn} with kwargs {kw};",
                      "#    streaming resamples the logits with plain torch trilinear (align_corners=False)."]
            if aniso_ratio > ANISO_THRESHOLD and (kw is None or kw.get('force_separate_z', None) is not False):
                lines += [f"#    The spacing is anisotropic (max/min = {aniso_ratio:.2f} > {ANISO_THRESHOLD}), where "
                          f"nnUNetv2_predict",
                          "#    resamples the low-resolution axis separately: expect deviations beyond boundary noise."]
            else:
                lines += ["#    At (near-)isotropic spacing this typically matches nnUNetv2_predict to within boundary "
                          "noise."]
        n_folds = len(self.list_of_parameters)
        if n_folds == 1:
            lines += ["#  * Model: single fold."]
        else:
            lines += [f"#  * Model: ensemble of {n_folds} folds (logits averaged per slab)."]
        if self._detached_head is not None:
            lines += [f"#  * Feature ring: {self.feature_ring_status[len('on: '):]} (--logit_ring keeps the logits)."]
        else:
            lines += [f"#  * Logit ring (feature ring {self.feature_ring_status})."]
        lines += ["#  * Output: segmentations (use nnUNetv2_predict if you need probabilities)."]
        if slab_axis is None:
            lines += ["#  * Slab axis: chosen per case to minimize VRAM (-slab_axis overrides)."]
        else:
            lines += [f"#  * Slab axis: {slab_axis} (set with -slab_axis)."]
        lines += ["#" * 92, ""]
        print("\n".join(lines))

    @torch.inference_mode()
    def predict_from_files_streaming(self,
                                     source_folder_or_list_of_lists: Union[str, List[List[str]]],
                                     output_folder: str,
                                     overwrite: bool = True,
                                     num_processes_preprocessing: Optional[int] = None,
                                     num_processes_export: Optional[int] = None,
                                     num_parts: int = 1,
                                     part_id: int = 0,
                                     slab_axis: Union[int, None] = None,
                                     ring_dtype: torch.dtype = torch.half):
        """
        Streaming counterpart of predict_from_files for a single model (no cascades, no probability export).

        Preprocessing runs in the persistent background workers, the GPU performs the fused sliding-window + resample +
        argmax pass, and writing the finished segmentations to disk is offloaded to the persistent export pool
        (predict_cases_streaming). The main process never holds more than one case's thin logit band.
        """
        with self._quiet():
            list_of_lists, output_filename_truncated, _ = self._manage_input_and_output_lists(
                source_folder_or_list_of_lists, output_folder, None, overwrite, part_id, num_parts, False)
            self._prepare_standalone_inference(num_cases=len(list_of_lists))
            self.print_streaming_deviation_notice(slab_axis)
            maybe_mkdir_p(output_folder)
            # store these so postprocessing / ensembling tooling can be pointed at the output folder, like nnU-Net does
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
            if len(list_of_lists) == 0:
                print('All requested cases are already predicted. Nothing to do.')
                return []
            return self.predict_cases_streaming(list_of_lists, None, output_filename_truncated,
                                                num_processes_preprocessing=num_processes_preprocessing,
                                                num_processes_export=num_processes_export, slab_axis=slab_axis,
                                                ring_dtype=ring_dtype)

    # ------------------------------------------------------------------ core ------------------------------------------
    def _choose_slab_axis(self, source_shape: List[int], patch_size: List[int], tile_step_size: float) -> int:
        """
        Pick the slab axis that minimizes the ring buffer, which dominates VRAM:
            ring ~ num_classes * ring_capacity(a) * internal_cross_section(a)
        with ring_capacity(a) = patch_size[a] + max_step_gap(a) + 8 (the actual ring is patch_size[a] + 1; this term
        is a ranking heuristic, and changing it changes the chosen axis - and with it the output - on some cases) and
        internal_cross_section(a) = prod(source_shape) / source_shape[a].
        num_classes and dtype are constant across axes, so they drop out of the argmin.

        For isotropic patches this reduces to "slab the longest axis" (smallest cross-section); for anisotropic
        patches it favors the thin-patch axis (small ring_capacity), which can be a large ring reduction.
        """
        total = float(np.prod(source_shape))
        steps = compute_steps_for_sliding_window(source_shape, patch_size, tile_step_size)
        best_axis, best_cost = 0, None
        for a in range(len(source_shape)):
            max_gap = max([steps[a][i + 1] - steps[a][i] for i in range(len(steps[a]) - 1)], default=0)
            ring_capacity = patch_size[a] + max_gap + 8
            cost = ring_capacity * (total / source_shape[a])
            if best_cost is None or cost < best_cost:
                best_cost, best_axis = cost, a
        return best_axis

    def _slab_coordinate_map(self, source_len: int, target_len: int, device: torch.device,
                             nearest_rule: Union[str, None] = None):
        """
        For each target index along the slab axis return (i0, i1, w) such that the resampled value is
        (1 - w) * source[i0] + w * source[i1]. Replicates torch F.interpolate(mode=linear, align_corners=False),
        or, with nearest_rule (the slab axis is the separate-z axis), copies one source slice: i0 == i1, w = 0.
        w stays on the host: it is float64 (MPS has no float64) and only read back as Python floats.
        """
        if nearest_rule is not None:
            idx = nearest_source_index(source_len, target_len, nearest_rule)
            return idx.to(device), idx.to(device), torch.zeros(target_len, dtype=torch.float64)
        scale = source_len / target_len
        j = torch.arange(target_len, dtype=torch.float64)
        src = torch.clamp((j + 0.5) * scale - 0.5, min=0.0)
        i0 = torch.clamp(torch.floor(src).long(), max=source_len - 1)
        i1 = torch.clamp(i0 + 1, max=source_len - 1)
        w = (src - i0.to(torch.float64))
        return i0.to(device), i1.to(device), w

    def _load_fold(self, fold_index: int):
        """
        Make the (possibly torch.compile'd) network hold fold fold_index's weights, unless it already does
        (nnUNetPredictor._load_parameters: tracked by identity of module and state dict, so a single model is loaded
        once and kept across cases). compile guards on input shape/stride, not weight values, so swapping weights does
        not recompile.
        """
        self._load_parameters(self.list_of_parameters[fold_index])

    def _place_fold_weights(self):
        """
        A 3d fold ensemble on CUDA swaps weights per slab, so its state dicts move to pinned host memory (faster copies;
        default) or, with fold_weights_on_device, to the device (+ one model of VRAM per fold). list_of_parameters is
        replaced once. A single model is loaded once and kept (_load_fold), and 2d keeps one network per fold, so both
        stay as they are.
        """
        params = self.list_of_parameters
        if len(params) < 2 or self.device.type != 'cuda' or len(self.configuration_manager.patch_size) != 3:
            return
        on_device = self.fold_weights_on_device
        if all(v.is_cuda if on_device else v.is_pinned() for p in params for v in p.values()):
            return
        self.list_of_parameters = [{k: v.to(self.device) if on_device else v.cpu().pin_memory() for k, v in p.items()}
                                   for p in params]

    def _fold_networks_2d(self) -> List[torch.nn.Module]:
        """
        2d fold ensembles: one copy of the network per fold, holding that fold's weights, on the device. Built on first
        use and kept across cases while the network and every state dict in list_of_parameters are the same objects
        (replacing one rebuilds them, as _load_fold reloads; in-place changes are not noticed).
        """
        target = unwrap_network(self.network)
        key = (target, *self.list_of_parameters)
        cached = self._fold_nets_2d
        if cached is None or len(cached[0]) != len(key) or any(a is not b for a, b in zip(cached[0], key)):
            self._fold_nets_2d = None                          # free the old copies before building new ones
            nets = []
            for params in self.list_of_parameters:
                net = copy.deepcopy(target)
                net.load_state_dict(params)
                nets.append(net.to(self.device).eval())
            cached = self._fold_nets_2d = (key, nets)
        return cached[1]

    @contextlib.contextmanager
    def _cache_emptied_after(self):
        """The public entry points run in this: the allocator cache is emptied once when they end (also on error), while
        the cores keep it across cases."""
        try:
            yield
        finally:
            empty_cache(self.device)

    def _mirror_predict(self, network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Mirror-augmented prediction with an explicit network instance (for resident-network ensembling)."""
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = network(x)
        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes do not match the input dimension'
            axes = [m + 2 for m in mirror_axes]
            combos = [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]
            for combo in combos:
                prediction = prediction + torch.flip(network(torch.flip(x, combo)), combo)
            prediction = prediction / (len(combos) + 1)
        return prediction

    def _gaussian(self) -> Union[torch.Tensor, None]:
        """The tile weighting on the device, or None without Gaussian weighting. compute_gaussian is cached, and
        warm_up and both streaming paths must hit the same cache entry, so this is the one place its arguments live."""
        if not self.use_gaussian:
            return None
        return compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                value_scaling_factor=10, device=self.device)

    def _seg_dtype(self) -> torch.dtype:
        """dtype of the segmentation on the device (uint8, int16 from 255 foreground labels on; see _revert_geometry)."""
        return torch.uint8 if len(self.label_manager.foreground_labels) < 255 else torch.int16

    def _streaming_kernels(self, ring_dtype: torch.dtype):
        """The fused Triton kernels (streaming_kernels) if they may be used here - CUDA, Triton importable, and their
        once-per-process self-test passed for this ring dtype - else None (torch path)."""
        if not self._use_triton_kernels or self.device.type != 'cuda':
            return None
        try:
            from nnunetv2.inference import streaming_kernels
        except ImportError:
            return None
        return streaming_kernels if streaming_kernels.kernels_selftest(self.device, ring_dtype) else None

    def _reduce_to_seg(self, resampled_logits: torch.Tensor, out_torch_dtype: torch.dtype) -> torch.Tensor:
        """Channel reduction: argmax for softmax (monotonic -> on logits), ordered threshold for sigmoid/regions."""
        lm = self.label_manager
        if lm.has_regions:
            seg = torch.zeros(resampled_logits.shape[1:], dtype=out_torch_dtype, device=resampled_logits.device)
            for region_idx, cls in enumerate(lm.regions_class_order):
                seg[resampled_logits[region_idx] > 0] = cls            # sigmoid(x) > 0.5 <=> x > 0
            return seg
        return resampled_logits.argmax(0).to(out_torch_dtype)          # softmax is monotonic -> argmax on logits

    def _revert_geometry(self, seg_xyz: np.ndarray, properties: dict) -> np.ndarray:
        """Revert cropping (insert into shape_before_cropping at bbox) and transpose_backward -> original geometry."""
        lm = self.label_manager
        out_dtype = np.uint8 if len(lm.foreground_labels) < 255 else np.uint16
        reverted = np.zeros(properties['shape_before_cropping'], dtype=out_dtype)
        reverted = insert_crop_into_image(reverted, seg_xyz, properties['bbox_used_for_cropping'])
        return reverted.transpose(self.plans_manager.transpose_backward)

    @torch.inference_mode()
    def _predict_and_export_streaming(self,
                                      data: torch.Tensor,
                                      properties: dict,
                                      slab_axis: Union[int, None] = None,
                                      ring_dtype: torch.dtype = torch.half):
        cm = self.configuration_manager
        patch_size = list(cm.patch_size)
        if len(patch_size) == 2:
            return self._predict_case_2d(data, properties, ring_dtype)
        assert len(patch_size) == 3, "streaming prediction supports 2d and 3d configurations only"

        device = self.device
        lm = self.label_manager
        num_classes = lm.num_segmentation_heads
        has_regions = lm.has_regions

        self.network = self.network.to(device).eval()
        num_folds = len(self.list_of_parameters)
        # feature ring (detach_head): the network returns its last features, the ring holds one block of them per fold
        # and the output layers are applied per finished source slice (head_logits)
        fr = self._detached_head
        if fr is not None and fr['network'] is not unwrap_network(self.network):
            fr = None
        if fr is not None:
            assert len(fr['heads']) == num_folds, 'list_of_parameters changed after detach_head'
            head_W, head_b = self._stacked_head(device)
            n_feat = fr['module'].in_channels

        autocast_ctx = torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context()
        with autocast_ctx:
            # ---- pad input to at least patch_size (sliding window requirement) ----
            data, slicer_revert_padding = pad_nd_image(data, patch_size, 'constant', {'value': 0}, True, None)
            # the whole input on the device (4 bytes per padded input voxel and channel): patches are cut there
            data = data.to(device)
            shape_padded = list(data.shape[1:])                                  # padded source shape (x, y, z)
            pad_before = [slicer_revert_padding[1 + ax].start for ax in range(3)]
            source_shape = [slicer_revert_padding[1 + ax].stop - slicer_revert_padding[1 + ax].start
                            for ax in range(3)]                                  # unpadded source (internal) shape

            # ---- target (resample) shape: this is what convert_..._with_correct_shape resamples to ----
            target_shape = list(properties['shape_after_cropping_and_before_resampling'])

            # ---- choose slab axis and the canonical (slab, cross0, cross1) ordering ----
            a = self._choose_slab_axis(shape_padded, patch_size, self.tile_step_size) \
                if slab_axis is None else int(slab_axis)
            cross = [ax for ax in range(3) if ax != a]
            perm = [a, cross[0], cross[1]]                  # canonical[i] == xyz[perm[i]]
            inv_perm = list(np.argsort(perm))               # canonical -> xyz

            P_slab = patch_size[a]
            cross_pad = (shape_padded[cross[0]], shape_padded[cross[1]])         # padded cross section size
            cross_src = (source_shape[cross[0]], source_shape[cross[1]])         # unpadded cross section size
            target_slab = target_shape[a]
            target_cross = (target_shape[cross[0]], target_shape[cross[1]])

            # ---- sliding window steps (padded source coords) ----
            steps = compute_steps_for_sliding_window(shape_padded, patch_size, self.tile_step_size)
            slab_steps = steps[a]
            ring_capacity = P_slab + self._ring_margin                         # thin band along slab axis

            if self.verbose:
                print(f'[streaming] source(padded) {shape_padded} unpadded {source_shape} target {target_shape}')
                print(f'[streaming] slab_axis={a} perm={perm} P_slab={P_slab} ring_capacity={ring_capacity} '
                      f'num_classes={num_classes} has_regions={has_regions}')

            # ---- GPU ring buffers (canonical layout: slab axis first) ----
            # logit ring: one channel per class; feature ring: fold k's features in channels k * n_feat ... + n_feat - 1
            ring_channels = num_classes if fr is None else num_folds * n_feat
            ring = torch.zeros((ring_channels, ring_capacity, *cross_pad), dtype=ring_dtype, device=device)
            ring_npred = torch.zeros((ring_capacity, *cross_pad), dtype=ring_dtype, device=device)
            if self.verbose:
                ring_mb = (ring.element_size() * ring.nelement() +
                           ring_npred.element_size() * ring_npred.nelement()) / 1024 ** 2
                print(f'[streaming] ring buffers = {ring_mb:.1f} MB '
                      f'({"logits" if fr is None else "features"} {tuple(ring.shape)} {ring_dtype})')

            gaussian = self._gaussian()
            gaussian_c = gaussian.permute(perm).contiguous() if self.use_gaussian else None

            # ---- output (canonical slab-first), single channel ----
            out_torch_dtype = self._seg_dtype()
            out_seg_canonical = torch.zeros((target_slab, *target_cross), dtype=out_torch_dtype, device=device)

            # ---- separate-z: the low-resolution axis copies its nearest source slice (None: trilinear) ----
            sep = self.separate_z_for_case(properties)
            slab_rule = sep[1] if sep is not None and sep[0] == a else None
            cross_sel = None                              # (cross dim 0/1, source index per target index)
            if sep is not None and sep[0] != a:
                k = cross.index(sep[0])
                cross_sel = (k, nearest_source_index(cross_src[k], target_cross[k], sep[1]).to(device))
            if self.verbose and sep is not None:
                print(f'[streaming] separate-z: nearest ({sep[1]}) along axis {sep[0]} '
                      f'({"slab" if slab_rule is not None else "cross-section"} axis)')

            # ---- slab-axis resampling coordinates (unpadded source -> target) ----
            i0_map, i1_map, w_map = self._slab_coordinate_map(source_shape[a], target_slab, device, slab_rule)
            pad_off = pad_before[a]
            cross0_crop = slice(pad_before[cross[0]], pad_before[cross[0]] + cross_src[0])
            cross1_crop = slice(pad_before[cross[1]], pad_before[cross[1]] + cross_src[1])

            # in-plane-resampled (normalized) source slices, keyed by unpadded source slab index
            inplane_cache = {}
            # host copies of the slab-axis maps: no device sync per target plane (identical values)
            i0_l, i1_l, w_l = i0_map.tolist(), i1_map.tolist(), w_map.tolist()

            # fused Triton kernels (CUDA + Triton, fp16 or fp32 ring, self-test passed)
            _sk = self._streaming_kernels(ring_dtype)
            use_kernels = _sk is not None
            fused_accumulate = use_kernels and self.use_gaussian
            fused_emit = use_kernels
            if fused_emit:
                sel = [None, None]
                if cross_sel is not None:
                    sel[cross_sel[0]] = cross_sel[1]
                tab_c0 = _sk.bilinear_tables(cross_src[0], target_cross[0], device, sel[0])
                tab_c1 = _sk.bilinear_tables(cross_src[1], target_cross[1], device, sel[1])
                region_labels = torch.tensor(lm.regions_class_order, dtype=torch.int32, device=device) \
                    if has_regions else None

            head_cache = {}

            def head_logits(src_idx_pad: int):
                """Feature ring: (W f + b n, n) of one finished source slice in fp32 (f = weighted feature sums of all
                folds, n = weight sum; _stacked_head), i.e. the logit ring's un-normalized slice. Cached for the (at
                most two) slices still in use."""
                cached = head_cache.get(src_idx_pad)
                if cached is None:
                    for k in [kk for kk in head_cache if kk < src_idx_pad - 1]:
                        del head_cache[k]
                    slot = src_idx_pad % ring_capacity
                    f = ring[:, slot, cross0_crop, cross1_crop].float()                # (K * F, cs0, cs1)
                    n = ring_npred[slot, cross0_crop, cross1_crop].float().contiguous()       # (cs0, cs1)
                    with torch.autocast(device.type, enabled=False):
                        lg = torch.mm(head_W, f.reshape(f.shape[0], -1)).view(head_W.shape[0], *n.shape)
                    lg += head_b[:, None, None] * n
                    cached = head_cache[src_idx_pad] = (lg, n)
                return cached

            def get_inplane(src_idx_unpadded: int) -> torch.Tensor:
                cached = inplane_cache.get(src_idx_unpadded)
                if cached is not None:
                    return cached
                slot = (src_idx_unpadded + pad_off) % ring_capacity
                if fr is None:
                    logit_slice = ring[:, slot, cross0_crop, cross1_crop]         # (C, cs0, cs1)
                    npred_slice = ring_npred[slot, cross0_crop, cross1_crop]      # (cs0, cs1)
                    normalized = (logit_slice / npred_slice).float()              # divide in ring dtype, then upcast
                else:
                    lg, n = head_logits(src_idx_unpadded + pad_off)
                    normalized = lg / n
                if cross_sel is not None:
                    # separate-z axis in the cross-section: select first, then bilinear is linear along the other
                    # cross axis and an exact identity along this one (its size already is the target size)
                    normalized = normalized.index_select(1 + cross_sel[0], cross_sel[1])
                resampled = F.interpolate(normalized[None], size=target_cross, mode='bilinear',
                                          align_corners=False)[0]                 # (C, tc0, tc1)
                inplane_cache[src_idx_unpadded] = resampled
                return resampled

            emitted = 0                     # target planes written so far
            released_pad = 0                # lowest padded source slice the ring must still hold

            def emit_available(final_hi_pad: int):
                nonlocal emitted, released_pad
                j = emitted
                while j < target_slab:
                    i0 = i0_l[j]; i1 = i1_l[j]
                    # padded source coords of the two neighbours
                    if max(i0, i1) + pad_off >= final_hi_pad:
                        break                                            # neighbour not finalized yet
                    w = w_l[j]
                    # lowest source slice the ring must still hold after this plane: the next plane's lower neighbour
                    released_pad = (i0_l[j + 1] + pad_off) if j + 1 < target_slab else 1 << 60
                    if fused_emit:
                        if fr is None:
                            s0 = (i0 + pad_off) % ring_capacity
                            s1 = (i1 + pad_off) % ring_capacity
                            l0, l1 = ring[:, s0, cross0_crop, cross1_crop], ring[:, s1, cross0_crop, cross1_crop]
                            n0, n1 = ring_npred[s0, cross0_crop, cross1_crop], ring_npred[s1, cross0_crop, cross1_crop]
                        else:                                            # fp32 logits of the two source slices
                            (l0, n0), (l1, n1) = head_logits(i0 + pad_off), head_logits(i1 + pad_off)
                        if region_labels is None:
                            _sk.emit_argmax(l0, l1, n0, n1, w, i1 != i0, tab_c0, tab_c1, out_seg_canonical[j])
                        else:
                            _sk.emit_regions(l0, l1, n0, n1, w, i1 != i0, tab_c0, tab_c1, region_labels,
                                             out_seg_canonical[j])
                        j += 1
                        continue
                    sl0 = get_inplane(i0)
                    sl1 = get_inplane(i1) if i1 != i0 else sl0
                    out_float = sl0 if i1 == i0 else (1.0 - w) * sl0 + w * sl1
                    out_seg_canonical[j] = self._reduce_to_seg(out_float, out_torch_dtype)
                    # evict in-plane slices below the current source index: source indices are monotone
                    # non-decreasing in j, so anything < i0 will never be needed again. Evicting per-slice (rather
                    # than per emit-pass) keeps the cache at ~2 entries instead of a whole pass' worth.
                    for k in [kk for kk in inplane_cache if kk < i0]:
                        del inplane_cache[k]
                    j += 1
                emitted = j

            # ---- sliding-window positions within a slab, patches cut from the device-resident input ----
            cross_offsets = [(o0, o1) for o0 in steps[cross[0]] for o1 in steps[cross[1]]]

            def cut_patch(slab_start: int, o0: int, o1: int) -> torch.Tensor:
                start = [0, 0, 0]
                start[a], start[cross[0]], start[cross[1]] = slab_start, o0, o1
                sl = (slice(None), *[slice(start[d], start[d] + patch_size[d]) for d in range(3)])
                return data[sl][None].clone(memory_format=torch.contiguous_format)

            zeroed_hi = 0

            def ensure_zeroed(upto: int):
                nonlocal zeroed_hi
                if upto <= zeroed_hi:
                    return
                # defensive: do not overwrite source slices that have not yet been emitted
                if upto - ring_capacity > released_pad and zeroed_hi > 0:
                    raise RuntimeError(f"ring_capacity={ring_capacity} too small: about to overwrite un-emitted "
                                       f"source slice (zeroing up to {upto}, still needed from {released_pad})")
                for _, rlo, rhi, _ in _ring_segments(zeroed_hi, upto, ring_capacity):
                    ring[:, rlo:rhi].zero_()
                    ring_npred[rlo:rhi].zero_()
                zeroed_hi = upto

            pcr0, pcr1 = patch_size[cross[0]], patch_size[cross[1]]

            def flush_slab(slab_start):
                # Accumulate all patches of this slab into the ring. For ensembling we loop folds OUTSIDE the
                # patch loop (one weight swap per fold per slab, not per patch). Accumulating n_predictions once
                # per fold makes the later logits / n_predictions an exact mean over folds (and overlapping patches).
                ensure_zeroed(slab_start + P_slab)
                for fold_index in range(num_folds):
                    self._load_fold(fold_index)
                    fold_ring = ring if fr is None else ring[fold_index * n_feat:(fold_index + 1) * n_feat]
                    for o0, o1 in cross_offsets:
                        pred = self._internal_maybe_mirror_and_predict(cut_patch(slab_start, o0, o1))[0]
                        pred_c = pred.permute(0, 1 + perm[0], 1 + perm[1], 1 + perm[2])
                        fused = fused_accumulate and pred_c.dtype == torch.half
                        if self.use_gaussian and not fused:
                            pred_c.mul_(gaussian_c)          # pred is ours alone: in place, no 2nd (C, patch) copy
                        for off, rlo, rhi, run in _ring_segments(slab_start, slab_start + P_slab, ring_capacity):
                            if fused:                        # ring += round16(pred * g), one pass
                                _sk.accumulate_(fold_ring[:, rlo:rhi, o0:o0 + pcr0, o1:o1 + pcr1],
                                                pred_c[:, off:off + run], gaussian_c[off:off + run])
                            else:
                                fold_ring[:, rlo:rhi, o0:o0 + pcr0, o1:o1 + pcr1] += pred_c[:, off:off + run]
                            if self.use_gaussian:
                                ring_npred[rlo:rhi, o0:o0 + pcr0, o1:o1 + pcr1] += gaussian_c[off:off + run]
                            else:
                                ring_npred[rlo:rhi, o0:o0 + pcr0, o1:o1 + pcr1] += 1
                        # free this patch's (C, *patch) output before the next forward: otherwise `pred` keeps it
                        # alive while the next one is computed (two outputs at the peak, 1.6 GB at 192^3 x 126 classes)
                        del pred, pred_c

            # Slab axis outermost: once a slab is flushed (all folds), the slices below the next one are final.
            for k, s_slab in enumerate(slab_steps):
                flush_slab(s_slab)
                emit_available(slab_steps[k + 1] if k + 1 < len(slab_steps) else shape_padded[a] + 1)
            assert emitted == target_slab, f"streaming emit incomplete: emitted {emitted} of {target_slab} target slices"

            if self.verbose and device.type == 'cuda':
                print(f'[streaming] peak GPU allocated this case = '
                      f'{torch.cuda.max_memory_allocated(device) / 1024 ** 3:.3f} GB')

            # no empty_cache per case: the next case reuses the cached blocks (_cache_emptied_after)
            del ring, ring_npred, inplane_cache, head_cache, data

        # ---- back to xyz order, revert cropping, revert transpose ----
        seg_xyz = np.transpose(out_seg_canonical.cpu().numpy(), inv_perm)         # (tx, ty, tz) == target_shape
        return self._revert_geometry(seg_xyz, properties)

    @torch.inference_mode()
    def _predict_case_2d(self, data: torch.Tensor, properties: dict, ring_dtype: torch.dtype = torch.half):
        """
        2D configurations: slices along axis 0 are independent and the slice axis is not resampled, so there is no
        ring buffer and no cross-slice machinery. Each slice runs its own 2D sliding window into a single-slice
        buffer, gets normalized + in-plane resampled + reduced to a segmentation, and is written out. Peak memory
        is one slice (C x in-plane). Fold ensembles run one resident network per fold (_fold_networks_2d).
        """
        assert self._detached_head is None, '2d configurations keep the logit ring (_feature_ring_applicable)'
        device = self.device
        lm = self.label_manager
        cm = self.configuration_manager
        patch_size = list(cm.patch_size)                                # (p0, p1)
        num_classes = lm.num_segmentation_heads
        num_folds = len(self.list_of_parameters)
        out_torch_dtype = self._seg_dtype()

        # Resident fold networks: a 2d slice has very few patches, so swapping weights per slice (like the 3d ring
        # path does per slab) would dominate runtime. 2d weights are small, so we keep one network per fold.
        self.network = self.network.to(device).eval()
        if num_folds == 1:
            self._load_fold(0)                                          # no-op if this model is loaded already
            fold_nets = [self.network]                                  # keep the (possibly compiled) network
        else:
            fold_nets = self._fold_networks_2d()

        autocast_ctx = torch.autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context()
        with autocast_ctx:
            data, slicer_revert_padding = pad_nd_image(data, patch_size, 'constant', {'value': 0}, True, None)
            data = data.to(device)                                      # patches are cut on the device, as in 3d
            shape_padded = list(data.shape[1:])                         # (S0, padded_y, padded_z)
            num_slices = shape_padded[0]
            cross0_crop = slice(slicer_revert_padding[2].start, slicer_revert_padding[2].stop)
            cross1_crop = slice(slicer_revert_padding[3].start, slicer_revert_padding[3].stop)

            target_shape = list(properties['shape_after_cropping_and_before_resampling'])
            assert target_shape[0] == num_slices, \
                (f"2d streaming assumes the slice axis is not resampled, but target slice count "
                 f"{target_shape[0]} != source {num_slices}.")
            target_cross = (target_shape[1], target_shape[2])
            # separate-z along the slice axis is the identity here (it is not resampled); in-plane it is a selection
            sep = self.separate_z_for_case(properties)
            cross_sel = None
            if sep is not None and sep[0] != 0:
                k = sep[0] - 1
                src_len = (cross0_crop, cross1_crop)[k].stop - (cross0_crop, cross1_crop)[k].start
                cross_sel = (k, nearest_source_index(src_len, target_cross[k], sep[1]).to(device))

            out_seg = torch.zeros((num_slices, *target_cross), dtype=out_torch_dtype, device=device)
            gaussian = self._gaussian()
            steps = compute_steps_for_sliding_window(shape_padded[1:], patch_size, self.tile_step_size)
            pcr0, pcr1 = patch_size[0], patch_size[1]
            grid = [(o0, o1) for o0 in steps[0] for o1 in steps[1]]
            if self.verbose:
                print(f'[streaming-2d] slices={num_slices} padded_cross={shape_padded[1:]} target_cross={target_cross} '
                      f'folds={num_folds} patch={patch_size} n_patches_per_slice={len(grid)}')

            for d in range(num_slices):
                slice_logits = torch.zeros((num_classes, *shape_padded[1:]), dtype=ring_dtype, device=device)
                slice_npred = torch.zeros(tuple(shape_padded[1:]), dtype=ring_dtype, device=device)
                for net in fold_nets:
                    for o0, o1 in grid:
                        patch = data[:, d, o0:o0 + pcr0, o1:o1 + pcr1][None].clone(
                            memory_format=torch.contiguous_format)
                        pred = self._mirror_predict(net, patch)[0]                    # (C, p0, p1)
                        if self.use_gaussian:
                            pred = pred * gaussian
                        slice_logits[:, o0:o0 + pcr0, o1:o1 + pcr1] += pred
                        if self.use_gaussian:
                            slice_npred[o0:o0 + pcr0, o1:o1 + pcr1] += gaussian
                        else:
                            slice_npred[o0:o0 + pcr0, o1:o1 + pcr1] += 1
                normalized = (slice_logits / slice_npred).float()[:, cross0_crop, cross1_crop]
                if cross_sel is not None:
                    normalized = normalized.index_select(1 + cross_sel[0], cross_sel[1])
                resampled = F.interpolate(normalized[None], size=target_cross, mode='bilinear',
                                          align_corners=False)[0]
                out_seg[d] = self._reduce_to_seg(resampled, out_torch_dtype)

        seg_xyz = out_seg.cpu().numpy()                                 # already (T0, T1, T2) == target_shape (xyz)
        return self._revert_geometry(seg_xyz, properties)

    # ------------------------------------------------------------------ validation helper ----------------------------
    @torch.inference_mode()
    def resample_argmax_reference_streaming(self, logits: torch.Tensor, target_shape: List[int],
                                            slab_axis: Union[int, None] = None,
                                            separate_z: Union[Tuple[int, str], None] = None) -> torch.Tensor:
        """
        separate_z: (low-resolution axis, index rule) as separate_z_for_case returns it, None = trilinear.
        Isolation helper for validation: takes already-finalized (normalized, unpadded) logits of shape
        (C, *source_shape) and produces the segmentation at ``target_shape`` using exactly the same per-slab
        streaming resample + reduction as the full pipeline. Returns the segmentation in xyz order (numpy).
        This lets us compare the streaming resampler against a full-volume F.interpolate baseline without the
        sliding-window reordering noise.
        """
        device = self.device
        source_shape = list(logits.shape[1:])

        # validation helper: any slab axis is correct (resampler is axis-agnostic), default to longest target
        a = int(np.argmax(target_shape)) if slab_axis is None else int(slab_axis)
        cross = [ax for ax in range(3) if ax != a]
        perm = [a, cross[0], cross[1]]
        inv_perm = list(np.argsort(perm))
        target_slab = target_shape[a]
        target_cross = (target_shape[cross[0]], target_shape[cross[1]])

        logits = logits.to(device)
        # canonical (C, slab, cross0, cross1)
        logits_c = logits.permute(0, 1 + perm[0], 1 + perm[1], 1 + perm[2]).contiguous()

        out_torch_dtype = self._seg_dtype()
        out_seg_canonical = torch.zeros((target_slab, *target_cross), dtype=out_torch_dtype, device=device)
        slab_rule = separate_z[1] if separate_z is not None and separate_z[0] == a else None
        cross_sel = None
        if separate_z is not None and separate_z[0] != a:
            k = cross.index(separate_z[0])
            cross_sel = (k, nearest_source_index(source_shape[separate_z[0]], target_cross[k], separate_z[1])
                         .to(device))
        i0_map, i1_map, w_map = self._slab_coordinate_map(source_shape[a], target_slab, device, slab_rule)

        inplane_cache = {}

        def get_inplane(idx):
            cached = inplane_cache.get(idx)
            if cached is not None:
                return cached
            sl = logits_c[:, idx].float()
            if cross_sel is not None:
                sl = sl.index_select(1 + cross_sel[0], cross_sel[1])
            resampled = F.interpolate(sl[None], size=target_cross, mode='bilinear', align_corners=False)[0]
            inplane_cache[idx] = resampled
            return resampled

        for j in range(target_slab):
            i0 = int(i0_map[j]); i1 = int(i1_map[j]); w = float(w_map[j])
            sl0 = get_inplane(i0)
            sl1 = get_inplane(i1) if i1 != i0 else sl0
            out_float = sl0 if i1 == i0 else (1.0 - w) * sl0 + w * sl1
            out_seg_canonical[j] = self._reduce_to_seg(out_float, out_torch_dtype)
            for k in [kk for kk in inplane_cache if kk < i0]:
                del inplane_cache[k]

        seg_canonical = out_seg_canonical.cpu().numpy()
        return np.transpose(seg_canonical, inv_perm)


def predict_entry_point_streaming():
    """CLI: nnUNetv2_predict_streaming - memory-efficient alternative to nnUNetv2_predict (single model)."""
    import argparse
    from batchgenerators.utilities.file_and_folder_operations import isdir
    from nnunetv2.utilities.file_path_utilities import get_output_folder

    parser = argparse.ArgumentParser(
        description='Memory-efficient STREAMING alternative to nnUNetv2_predict. It fuses sliding-window prediction, '
                    'resampling and segmentation export into a single streaming pass along one axis, so the full '
                    '(num_classes, *image_shape) probability volume never has to be held in memory. This is ideal for '
                    'datasets with many classes and/or very large images. Supports 2d and 3d configurations and fold '
                    'ensembling. NOTE: no probability export. Predicted logits are resampled with torch trilinear, '
                    'following the plans\' separate-z path where it fires (bilinear in-plane + nearest along the '
                    'low-resolution axis); resampling it cannot reproduce falls back to trilinear (a notice is '
                    'printed at runtime).')
    parser.add_argument('-i', type=str, required=True,
                        help='Input folder. Use the correct channel numberings for your files (_0000 etc). File '
                             'endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. Dataset name or id.')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='nnU-Net trainer class used for training. Default: nnUNetTrainer. If the trainer class '
                             'cannot be imported (e.g. a custom trainer), the network is built with the base trainer '
                             '(safe whenever the trainer does not change the architecture).')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction.')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Folds of the trained model to use. Multiple folds are ensembled (logits averaged per '
                             'slab). Default: 0 1 2 3 4')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. Default: 0.5.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Disable test time augmentation (mirroring). Faster, less accurate. The mirror axes '
                             'themselves are always taken from the checkpoint.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output (prints buffer sizes, slab axis, peak VRAM, etc.).')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files).')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.safetensors',
                        help='Name of the checkpoint to use. Default: checkpoint_final.safetensors. If it does not '
                             'exist, the same name with .pth is used (models trained with older nnU-Net versions have '
                             '.pth checkpoints), and vice versa.')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export (writing). Default: 3')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict_streaming calls you will make. Default: 1')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='Which part is this (0 .. num_parts-1)? You are responsible for putting parts on '
                             'separate GPUs via CUDA_VISIBLE_DEVICES.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Device for inference: 'cuda', 'cpu' or 'mps'. Do NOT set the GPU id here, use "
                             "CUDA_VISIBLE_DEVICES=X instead.")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Disable progress bar (recommended for non-interactive HPC jobs).')
    parser.add_argument('-slab_axis', type=int, required=False, default=-1,
                        help='Spatial axis (0/1/2) to stream along. Default: -1 = auto (chosen to minimize VRAM).')
    parser.add_argument('--force_trilinear', action='store_true', required=False, default=False,
                        help='Resample plain trilinear even where the plans\' separate-z path would fire.')
    parser.add_argument('--compile', choices=('on', 'off'), required=False, default=None,
                        help='torch.compile the network. Default: follow the nnUNet_compile environment variable, '
                             'else compile when predicting more than 4 cases, wherever available (CUDA + Triton). '
                             'Compiling costs a few seconds on the first case (~4 s with a cold cache; the cache '
                             'persists: TORCHINDUCTOR_CACHE_DIR, default ~/.cache/nnunetv2/torchinductor) and saves '
                             '7-17 %% per case afterwards, so it does not pay for a single case.')
    parser.add_argument('--fp32_ring', action='store_true', required=False, default=False,
                        help='Accumulate the sliding-window ring buffer in float32 instead of float16 (float16 is '
                             'also what nnUNetPredictor accumulates its logits in). Uses ~2x the ring VRAM.')
    parser.add_argument('--logit_ring', action='store_true', required=False, default=False,
                        help='Keep the class logits in the ring buffer. By default (3d, standard UNets) the ring holds '
                             'the decoder\'s last features instead (32 channels per fold instead of one per class) and '
                             'the network\'s output layer is applied per output slice: same segmentation up to float '
                             'rounding, less VRAM, slightly faster. Fold ensembles use it only while folds x 32 is '
                             'below the class count.')
    parser.add_argument('--fold_weights_on_device', action='store_true', required=False, default=False,
                        help='Fold ensembles (3d): keep all folds\' weights on the GPU instead of in pinned host '
                             'memory. Faster weight swaps (they happen per slab), costs one model of VRAM per fold. No '
                             'effect for a single fold or 2d configurations.')

    print("\n#######################################################################\n"
          "Please cite the following papers when using nnU-Net's streaming inference:\n"
          "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
          "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
          "Nature methods, 18(2), 203-211.\n"
          f"{STREAMING_CITATION}\n#######################################################################\n")

    args = parser.parse_args()
    folds = [i if i == 'all' else int(i) for i in args.f]

    assert args.part_id < args.num_parts, 'part_id must be smaller than num_parts (they are 0-indexed).'
    assert args.device in ['cpu', 'cuda', 'mps'], \
        f'-device must be cpu, mps or cuda. Got: {args.device}.'
    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # the predictor enables expandable allocator segments itself (maybe_enable_expandable_segments)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    predictor = nnUNetPredictorStreaming(tile_step_size=args.step_size,
                                         use_gaussian=True,
                                         use_mirroring=not args.disable_tta,
                                         perform_everything_on_device=True,
                                         device=device,
                                         verbose=args.verbose,
                                         verbose_preprocessing=args.verbose,
                                         allow_tqdm=not args.disable_progress_bar,
                                         force_trilinear=args.force_trilinear,
                                         compile=None if args.compile is None else args.compile == 'on',
                                         feature_ring=not args.logit_ring,
                                         fold_weights_on_device=args.fold_weights_on_device)
    # use the normal initialization; fall back to building the net with the base trainer if the (possibly custom)
    # trainer class named in the checkpoint cannot be imported.
    try:
        predictor.initialize_from_trained_model_folder(model_folder, folds, checkpoint_name=args.chk)
    except RuntimeError as e:
        if 'trainer' in str(e).lower():
            print(f'Could not import the trainer class named in the checkpoint ({e}).\n'
                  f'Falling back to building the network with the base nnUNetTrainer. This is correct as long as the '
                  f'trainer variant does not change the architecture (load_state_dict would fail otherwise).')
            predictor.initialize_from_folder_base_trainer(model_folder, folds, checkpoint_name=args.chk)
        else:
            raise

    predictor.predict_from_files_streaming(
        args.i, args.o,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_export=args.nps,
        num_parts=args.num_parts,
        part_id=args.part_id,
        slab_axis=None if args.slab_axis < 0 else args.slab_axis,
        ring_dtype=torch.float32 if args.fp32_ring else torch.half,
    )


if __name__ == '__main__':
    predict_entry_point_streaming()
