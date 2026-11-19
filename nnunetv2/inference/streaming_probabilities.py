"""
Probability export for nnUNetPredictorStreaming (opt-in: save_probabilities).

Streaming inference never builds the (C, *image shape) probability volume, neither at the network's resolution nor at
the image's. With save_probabilities each finished output plane's class probabilities (label_manager's inference
nonlinearity: softmax, or sigmoid for region models) are converted on the device, gathered a few planes at a time and
written by a background thread into an on-disk blosc2 array. The full volume only ever exists on disk. Without
save_probabilities none of this runs.

File format: <output>.b2nd, next to <output>.pkl (the case's properties, as nnUNetv2_predict writes them)
  - a blosc2 NDArray of shape (C, *image shape): channels (classes, or regions for region models) first, then the image
    axes in the order of the model's reader, i.e. the layout of the 'probabilities' array in the .npz files that
    nnUNetv2_predict --save_probabilities writes. Outside the cropped region it holds what nnU-Net exports there
    (LabelManager.revert_cropping_on_probabilities: background 1 for softmax models, 0 for region models).
  - dtype uint16 by default, value = round(p * 65535). The quantization error is at most 1 / 131070 (7.6e-6)
    everywhere, which is far below the numerical noise of the prediction itself (half-precision network outputs and
    accumulation). Fixed point turns everything below 7.6e-6 into an exact 0 (and confident interiors into exactly
    65535), so the files compress 150-450x with 100+ classes, at ~10 GB/s with blosc2 >= 4.15.
  - dtype float32 on request (probability dtype 'float32', --float32): the probabilities themselves, lossless. Floats
    keep the tiny probabilities every class has far away from its structure, and their mantissas are noise: float32
    compresses only 1.3x, so files are 14-36 GB per CT with 117-125 classes (46-57 MB as uint16), writing them makes
    the prediction 2.5-4x slower (RTX 4090, local NVMe), and the device buffers are twice as large.
  - No float16. It would be worse than uint16 on every count that matters for probabilities: its absolute error near 1
    is up to 2.4e-4 (uint16: 7.6e-6 everywhere), and since it still stores tiny probabilities (down to ~6e-8) with
    noisy mantissas it compresses ~9x instead of ~55x (26-class test volume; uint16 150-450x with 100+ classes). Its
    only advantage, relative precision for tiny probabilities, is what float32 is for.
  - chunks: a few planes along the axis the prediction streams along x the full cross-section x as many channels as
    make ~8 MiB; blocks hold one channel (<= 256 KiB), so reading one channel decompresses only that channel. ZSTD
    level 1 with byte shuffle (float32: + bytedelta). Each write covers whole chunks (no recompression of partly
    written chunks). Chunks of one channel would make writes 2-3x slower: blosc2 spends ~0.1 ms per chunk, however
    small.
  - torch < 2.3 has no torch.uint16: there the staging rows are int16 holding the same 16 bits (values wrapped through
    int32 on the device, reinterpreted as uint16 on the host), so the file is the same.
  - vlmeta['nnunet']: {'format': ..., 'version': 1, 'dtype': 'uint16' | 'float32', 'scale': 65535 (uint16) or 1
    (float32), 'nonlinearity': 'softmax' | 'sigmoid'}; probability = stored value / scale.
Read with open_streamed_probabilities (the blosc2 array, values as stored) or load_streamed_probabilities (float32
numpy, the whole volume in RAM, divided by the scale).
"""
import os
import queue
import threading
import warnings
from typing import List, Sequence, Union

import blosc2
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import save_pickle

from nnunetv2.training.dataloading.foreground_locations import MMAP_KWARGS
from nnunetv2.utilities.label_handling.label_handling import LabelManager

PROBABILITIES_FILE_ENDING = '.b2nd'
PROBABILITY_SCALE = 65535
FORMAT_NAME = 'nnU-Net streamed probabilities'
FORMAT_VERSION = 1
PLANES_PER_CHUNK = 4            # output planes per chunk row (= per write)
COMPRESSION_THREADS = 8
NUM_ROWS = 2                    # staging rows (and host rows) that take turns; 4 or 6 were not faster on an RTX 4090
TARGET_CHUNK_BYTES = 8 * 2 ** 20
MAX_BLOCK_BYTES = 2 ** 18
# storage dtypes: numpy dtype, stored value of p = 1 (vlmeta 'scale'), blosc2 filters; the torch dtype of the same name
# is looked up when a writer is made
PROBABILITY_DTYPES = {
    'uint16': (np.uint16, PROBABILITY_SCALE, [blosc2.Filter.SHUFFLE]),
    'float32': (np.float32, 1, [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]),
}
# torch.uint16 exists from torch 2.3; without it uint16 rows are staged as int16 (same bits, see the module docstring)
TORCH_HAS_UINT16 = hasattr(torch, 'uint16')
FLOAT32_WARNING = (
    "Probabilities are saved losslessly as float32. Float probabilities hardly compress (1.3x, against 150-450x for "
    "the default uint16 with 100+ classes): expect files of up to num_classes x voxels x 4 bytes (e.g. 14-36 GB per "
    "whole-body CT with 117-125 classes, against 46-57 MB as uint16), a prediction several times slower (2.5-4x there "
    "on an RTX 4090) and twice the GPU buffer memory. The default (uint16, error <= 7.6e-6) is far below the numerical "
    "noise of the prediction itself.")


def check_probability_dtype(dtype: str):
    if dtype not in PROBABILITY_DTYPES:
        raise ValueError(f'probability dtype must be one of {list(PROBABILITY_DTYPES)}, got {dtype!r}')


def chunks_and_blocks(shape: Sequence[int], slab_file_axis: int, planes_per_chunk: int, itemsize: int):
    """Chunks and blocks of the probability file of shape (C, *image shape): chunks span planes_per_chunk planes along
    the slab axis (spatial index slab_file_axis), the full cross-section and ~TARGET_CHUNK_BYTES worth of channels;
    blocks one channel, C-order contiguous within the chunk, at most MAX_BLOCK_BYTES."""
    spatial = [int(i) for i in shape[1:]]
    spatial[slab_file_axis] = min(int(planes_per_chunk), spatial[slab_file_axis])
    per_channel = int(np.prod(spatial)) * itemsize
    channels = int(min(shape[0], max(1, TARGET_CHUNK_BYTES // per_channel)))
    blocks = list(spatial)
    for ax in range(3):                 # shrink from the outermost spatial axis until a block fits
        inner = int(np.prod(blocks[ax + 1:])) * itemsize
        blocks[ax] = int(max(1, min(blocks[ax], MAX_BLOCK_BYTES // inner)))
        if int(np.prod(blocks)) * itemsize <= MAX_BLOCK_BYTES:
            break
    return (channels, *spatial), (1, *blocks)


def outside_crop_probabilities(label_manager: LabelManager) -> torch.Tensor:
    """Per channel, what nnU-Net exports outside the cropped region: LabelManager.revert_cropping_on_probabilities on a
    1-voxel image with an empty crop (so a label manager that overrides it is followed)."""
    C = label_manager.num_segmentation_heads
    return label_manager.revert_cropping_on_probabilities(torch.zeros((C, 0, 0, 0)), [[0, 0]] * 3,
                                                          (1, 1, 1))[:, 0, 0, 0].float()


class StreamingProbabilityWriter:
    """
    Writes one case's probabilities while the streaming pass emits its output planes in order along the slab axis.

    Geometry: the streaming pass works in the plans' axis order (after transpose_forward) on the cropped region; plane j
    of the slab axis `slab_axis` covers the cropped cross-section. The file is in the reader's axis order and covers the
    uncropped image. The writer keeps NUM_ROWS staging rows of one chunk row each (planes_per_chunk planes along the
    slab axis, full cross-section, in the file's axis order; on the device on CUDA, else on the host). put() applies the
    nonlinearity, converts and inserts one plane; a full row is copied into a host row (on CUDA: pinned, on a side
    stream) and written by a background thread. The rows take turns, so the device keeps working while rows are copied
    and compressed; a staging row waits (on the device) for its previous copy, a host row for the writer.
    """

    def __init__(self, output_file_truncated: str, label_manager: LabelManager, properties: dict,
                 transpose_backward: Sequence[int], device: torch.device, planes_per_chunk: int = PLANES_PER_CHUNK,
                 compression_threads: int = COMPRESSION_THREADS, dtype: str = 'uint16'):
        """Nothing is allocated or written before begin(slab_axis), which the streaming pass calls. The caller ends the
        writer with close() after the pass, or abort() if the pass failed. dtype: 'uint16' (default) or 'float32'."""
        check_probability_dtype(dtype)
        if dtype == 'float32':
            warnings.warn(FLOAT32_WARNING, stacklevel=2)
        self.path = output_file_truncated + PROBABILITIES_FILE_ENDING
        self.tmp_path = output_file_truncated + '.partial' + PROBABILITIES_FILE_ENDING
        self.pkl_path = output_file_truncated + '.pkl'
        self.label_manager = label_manager
        self.properties = properties
        self.transpose_backward = [int(i) for i in transpose_backward]
        self.device = device
        self.K = int(planes_per_chunk)
        self.compression_threads = max(1, int(compression_threads))
        self.dtype = dtype
        self.np_dtype, self.scale, self.filters = PROBABILITY_DTYPES[dtype]
        # torch < 2.3 (no torch.uint16): int16 rows with the same bits (_to_stored, _write_loop)
        self.int16_bits = dtype == 'uint16' and not TORCH_HAS_UINT16
        self.torch_dtype = torch.int16 if self.int16_bits else getattr(torch, dtype)
        self.fill = outside_crop_probabilities(label_manager) * self.scale
        self.array = None
        self.thread = None
        self.copy_stream = None
        self.staging = self.host = None

    @torch.inference_mode()
    def begin(self, slab_axis: int):
        """Create the (partial) file, the rows and the writer thread; slab_axis in the plans' axis order."""
        assert self.array is None, 'begin() was called before'
        full = [int(i) for i in self.properties['shape_before_cropping']]       # plans' axis order
        bbox = [[int(lo), int(hi)] for lo, hi in self.properties['bbox_used_for_cropping']]
        tb = self.transpose_backward
        a = int(slab_axis)
        self.offset, self.end = bbox[a]              # file rows (slab axis) of plane 0 and past the last plane
        self.length = full[a]                        # planes along the slab axis in the file
        self.d = tb.index(a)                         # file axis of the slab axis (spatial index)
        # the two other file axes and the plans' axes they hold, in file order
        others = [i for i in range(3) if i != self.d]
        self.crop = tuple(slice(*bbox[tb[i]]) for i in others)
        cross = [ax for ax in range(3) if ax != a]
        self.transpose_plane = [tb[i] for i in others] != cross                # plane (C, cross0, cross1) -> file order
        shape = (self.label_manager.num_segmentation_heads, *[full[tb[i]] for i in range(3)])
        chunks, blocks = chunks_and_blocks(shape, self.d, self.K, np.dtype(self.np_dtype).itemsize)
        self.row_shape = list(shape)
        self.row_shape[1 + self.d] = self.K
        cparams = {'codec': blosc2.Codec.ZSTD, 'clevel': 1, 'filters': self.filters,
                   'filters_meta': [0] * len(self.filters), 'nthreads': self.compression_threads}
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)
        self.array = blosc2.empty(shape, dtype=self.np_dtype, chunks=chunks, blocks=blocks, urlpath=self.tmp_path,
                                  mode='w', cparams=cparams)

        self.cuda = self.device.type == 'cuda'
        cpu = torch.device('cpu')
        # elsewhere (CPU, MPS: uint16 support is incomplete there) the planes are converted on the host
        self.staging = [self._row(self.device if self.cuda else cpu) for _ in range(NUM_ROWS)]
        for row in self.staging:                    # the cross-section outside the crop is never written again
            self._fill(row)
        self.host = [self._row(cpu, pin=self.cuda) for _ in range(NUM_ROWS)]
        self.copy_stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.copied = [None] * NUM_ROWS              # CUDA events: staging[i] has been copied to host[i]
        self.host_free = [threading.Semaphore(1) for _ in range(NUM_ROWS)]
        self.cur = 0                                 # staging row being filled
        self.row0 = None                             # file row (slab axis) where the current chunk row starts
        self.written = 0                             # file rows [0, written) handed to the writer thread
        self.error = None
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._write_loop, name='nnunet_probability_writer', daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ rows -----------------------------------------
    def _row(self, device: torch.device, pin: bool = False) -> torch.Tensor:
        return torch.empty(self.row_shape, dtype=self.torch_dtype, device=device, pin_memory=pin)

    def _to_stored(self, values: torch.Tensor) -> torch.Tensor:
        """Stored values (float, e.g. 0..65535 for uint16) in the dtype of the rows; for int16 rows (torch < 2.3) they
        wrap through int32 into the same 16 bits that uint16 would hold."""
        if self.int16_bits:
            return values.to(torch.int32).to(torch.int16)
        return values.to(self.torch_dtype)

    def _fill(self, row: torch.Tensor):
        row.copy_(self._to_stored(self.fill.to(row.device)).view(-1, 1, 1, 1).expand_as(row))

    def _rows(self, start: int, stop: int) -> tuple:
        sl = [slice(None)] * 4
        sl[1 + self.d] = slice(start, stop)
        return tuple(sl)

    def _start_row(self, row0: int):
        i = self.cur
        if self.copied[i] is not None:               # its previous row must have left for the host
            torch.cuda.current_stream(self.device).wait_event(self.copied[i])
        if row0 < self.offset or row0 + self.K > self.end:
            self._fill(self.staging[i])              # planes outside the crop along the slab axis
        self.row0 = row0

    def _fill_rows_until(self, stop: int):
        """Whole chunk rows outside the crop along the slab axis, up to file row stop."""
        for r in range(self.written, stop, self.K):
            self._start_row(r)
            self._flush_row()

    def _flush_row(self):
        i = self.cur
        start, stop = self.row0, min(self.row0 + self.K, self.length)
        self.host_free[i].acquire()                  # the writer is done with host[i] (backpressure: NUM_ROWS rows)
        event = None
        if self.cuda:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(ready)
                self.host[i].copy_(self.staging[i], non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.copy_stream)
            self.copied[i] = event
        else:
            self.host[i].copy_(self.staging[i])
        self.queue.put((i, start, stop, event))
        self.cur = (i + 1) % NUM_ROWS
        self.written = stop
        self.row0 = None

    # ------------------------------------------------------------------ writer thread --------------------------------
    def _write_loop(self):
        while (item := self.queue.get()) is not None:
            i, start, stop, event = item
            try:
                if self.error is None:
                    if event is not None:
                        event.synchronize()
                    # a partial row (the image ends inside it) is a strided view: blosc2 wants C order
                    src = self.host[i].numpy()[self._rows(0, stop - start)]
                    if self.int16_bits:
                        src = src.view(np.uint16)
                    self.array[self._rows(start, stop)] = np.ascontiguousarray(src)
            except BaseException as e:            # reported to the predicting thread at its next put / close
                self.error = e
            finally:
                self.host_free[i].release()

    def _check_error(self):
        if self.error is not None:
            raise RuntimeError(f'writing the probabilities to {self.tmp_path} failed') from self.error

    # ------------------------------------------------------------------ API ------------------------------------------
    @torch.inference_mode()
    def put(self, j: int, logits: torch.Tensor):
        """Plane j (0-based along the slab axis, in emission order) of the cropped region: logits (C, cross0, cross1),
        float32, in the plans' axis order of the two cross axes."""
        self._check_error()
        g = self.offset + j
        row0 = g - g % self.K
        if self.row0 is not None and row0 != self.row0:
            self._flush_row()
        if self.row0 is None:
            self._fill_rows_until(row0)
            self._start_row(row0)
        probs = self.label_manager.apply_inference_nonlin(logits)
        if self.dtype == 'uint16':
            probs = self._to_stored(probs.mul_(PROBABILITY_SCALE).round_())
        if not self.cuda:
            probs = probs.cpu()
        if self.transpose_plane:
            probs = probs.transpose(1, 2)
        self.staging[self.cur].select(1 + self.d, g - row0)[(slice(None), *self.crop)] = probs

    @torch.inference_mode()
    def close(self):
        """Write the remaining rows, wait for the writer, write the case's properties (<output>.pkl, as nnUNetv2_predict
        does) and finalize the file (rename from its .partial name). Aborts itself if any of that fails."""
        assert self.array is not None, 'the streaming pass did not begin() this writer'
        try:
            if self.row0 is not None:
                self._flush_row()
            self._fill_rows_until(self.length)
            self.queue.put(None)
            self.thread.join()
            self._check_error()
            self.array.schunk.vlmeta['nnunet'] = {
                'format': FORMAT_NAME, 'version': FORMAT_VERSION, 'dtype': self.dtype, 'scale': self.scale,
                'nonlinearity': 'sigmoid' if self.label_manager.has_regions else 'softmax'}
            self.array = None                       # closes the file
            save_pickle(self.properties, self.pkl_path)
            os.replace(self.tmp_path, self.path)
        except BaseException:
            self.abort()
            raise
        self._release_buffers()

    def abort(self):
        """Stop the writer and delete the partial file (an exception during or after the streaming pass)."""
        if self.thread is not None and self.thread.is_alive():
            self.error = self.error or RuntimeError('aborted')
            self.queue.put(None)
            self.thread.join()
        self.array = None
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)
        self._release_buffers()

    def _release_buffers(self):
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        self.staging = self.host = None


# ---------------------------------------------------------------------- reading -------------------------------------
def open_streamed_probabilities(path: str) -> blosc2.NDArray:
    """The probability file as a blosc2 array (read-only, memory-mapped where supported), values as stored: divide by
    vlmeta['nnunet']['scale'] (65535 for the default uint16 files, 1 for float32 files) for probabilities. Slicing
    reads only what is needed, e.g. arr[c] = one class."""
    return blosc2.open(path, mode='r', **MMAP_KWARGS)


def load_streamed_probabilities(path: str, channels: Union[None, int, List[int]] = None) -> np.ndarray:
    """Probabilities as float32 (C, *image shape), or of the given channel(s) only. The whole selection is held in
    memory: for 100+ classes and large images that is tens of GB; slice open_streamed_probabilities instead."""
    arr = open_streamed_probabilities(path)
    inv_scale = np.float32(1.0 / arr.schunk.vlmeta['nnunet']['scale'])
    single = isinstance(channels, (int, np.integer))
    channels = list(range(arr.shape[0])) if channels is None else [int(channels)] if single else list(channels)
    out = np.empty((len(channels), *arr.shape[1:]), dtype=np.float32)
    for k, c in enumerate(channels):
        np.multiply(arr[c], inv_scale, out=out[k])
    return out[0] if single else out
