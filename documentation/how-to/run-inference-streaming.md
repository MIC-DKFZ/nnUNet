# Memory-efficient (streaming) inference

`nnUNetv2_predict_streaming` is an **optional, drop-in alternative to [`nnUNetv2_predict`](run-inference.md)**.
It produces the same segmentations up to floating-point rounding at label boundaries (see below), but is designed
for cases where the default predictor needs an impractical amount of RAM/VRAM: datasets with **many classes** and/or
**very large images** (e.g. 100+ classes at 512×512×1024).

You do not need this for normal datasets. Use the regular [`nnUNetv2_predict`](run-inference.md) unless memory
is the bottleneck.

## Citation

If you use streaming inference (`nnUNetv2_predict_streaming`, `nnUNetPredictorStreaming` or the streaming validation of
the `_SV` trainers), please cite the following paper in addition to nnU-Net:

```text
Isensee, F., Wasserthal, J., Shiyam Sundar, L. K., Rokuss, M., Ulrich Harsy, C., Langenberg, M., Happold, S. M., &
Maier-Hein, K. H. (2026). Segment Everything, Materialize Nothing: Exact Streaming Inference for Many-Class 3D
Segmentation. arXiv:XXXX.XXXXX.
```

## Why it exists

The default predictor materializes the full `(num_classes, *image_shape)` probability volume, resamples that whole
volume to the original resolution, and only then runs argmax. For 120 classes at 512×512×1024 the resampled volume
alone is >100 GB.

`nnUNetv2_predict_streaming` instead **fuses sliding-window prediction, resampling and segmentation export into a
single streaming pass** along one spatial axis. It keeps only a thin band (a "slab") of the prediction in GPU memory
at any time and writes the segmentation incrementally, so the full probability volume is never held in memory. The
slab axis is chosen automatically to minimize VRAM.

## How it differs from `nnUNetv2_predict`

Segmentations match `nnUNetv2_predict` up to floating-point rounding at label boundaries; they are not bit-for-bit
identical. A notice printed at startup summarizes how the run predicts. In detail:

- **Resampling follows the plans where it can.** Predicted logits are resampled with torch trilinear
  interpolation (`F.interpolate(mode='trilinear', align_corners=False)`), computed separably slab by slab.
  - For plans that resample probabilities with plain torch trilinear (`resampling_fn_probabilities` =
    `resample_torch_*` with `force_separate_z=False`, e.g. the ResEnc/`torchres` presets), results match
    `nnUNetv2_predict` to within boundary-level floating-point/GPU noise.
  - For plans that resample separately along a low-resolution axis (`resample_torch_fornnunet` or the default
    `resample_data_or_seg_to_shape` with order 1 / `order_z` 0, `force_separate_z` None or True), streaming takes
    the same separate-z path for the same cases: bilinear in-plane, nearest along the low-resolution axis,
    decided per case exactly as `nnUNetv2_predict` decides it. Results match to within the same boundary noise
    (the scipy resampler computes in float64, streaming in float32).
  - Resampling streaming cannot reproduce (e.g. cubic probability resampling) is replaced by plain trilinear;
    the runtime notice says so. `--force_trilinear` uses plain trilinear everywhere.
- **Probabilities go to `.b2nd` files, not `.npz`.** `--save_probabilities` writes `<case>.b2nd` (+ `<case>.pkl`)
  next to each segmentation, slice by slice during the prediction (see [Probability export](#probability-export)).
  `nnUNetv2_ensemble` does not read these files; to ensemble configurations with it, use `nnUNetv2_predict`.
- **Single configuration**, but **fold ensembling is supported** (logits are averaged per slab). 2d and 3d
  configurations are both supported. Cascaded models (previous-stage predictions) are not supported.
- **Region-based models are supported** (sigmoid outputs, `regions_class_order` in `dataset.json`, see
  [region-based training](../region_based_training.md)). The segmentation follows `nnUNetv2_predict`: each
  region is thresholded at sigmoid 0.5 and the regions are painted in `regions_class_order`, so a later region
  overwrites an earlier one. Streaming thresholds the resampled logits at 0 instead, which is the same decision without
  computing the sigmoid.

Segmentation quality is otherwise equivalent — validated against `nnUNetv2_predict` at >99.999% voxel agreement for
2d and 3d torchres models with 2-fold ensembling, with the few differing voxels being boundary-level fp/GPU noise.

## Usage

The interface mirrors `nnUNetv2_predict`:

```bash
nnUNetv2_predict_streaming -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
```

Fold ensembling (multiple folds are averaged):

```bash
nnUNetv2_predict_streaming -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION -f 0 1 2 3 4
```

Most `nnUNetv2_predict` flags are accepted (`-p`, `-tr`, `-f`, `-step_size`, `--disable_tta`, `-chk`,
`--continue_prediction`, `-npp`, `-nps`, `-num_parts`, `-part_id`, `-device`, `--disable_progress_bar`, `--verbose`).
Test-time augmentation uses the mirror axes stored in the checkpoint, exactly like `nnUNetv2_predict`.

Streaming-specific options:

- `--save_probabilities`: also write the predicted probabilities, see [Probability export](#probability-export).
- `--float32` (with `--save_probabilities`): store the probabilities losslessly as float32 instead of `uint16`. Much
  larger files and a much slower prediction; see [Probability export](#probability-export).
- `-slab_axis`: spatial axis (0/1/2) to stream along. Default: auto (chosen to minimize VRAM).
- `--compile {on,off}`: `torch.compile` the network. Default: follow the `nnUNet_compile` environment variable if it is
  set, otherwise compile when more than 4 cases are predicted and torch.compile is available (CUDA with Triton; not
  on CPU, MPS or Windows), and run a single image or a few without compiling. `--compile` overrides the environment
  variable; `--compile on` where compile is unavailable is an error. Compiling costs a few seconds on the first case
  (~4 s with a cold compile cache, ~3 s with a warm one) and makes every forward pass faster afterwards, so it pays
  over several cases. The cache persists in `TORCHINDUCTOR_CACHE_DIR` (default `~/.cache/nnunetv2/torchinductor`).
- `--fp32_ring`: accumulate the sliding-window buffer in float32 instead of float16 (≈2× the buffer VRAM; rarely
  needed). Works with both buffer layouts below.
- `--logit_ring`: keep one buffer channel per class instead of the network's last features (see *Feature ring* below).
  Same segmentation up to floating-point rounding; uses more VRAM.
- `--fold_weights_on_device`: fold ensembles (3d) keep every fold's weights on the GPU instead of in pinned host memory.
  The weights are swapped once per slab, so this is faster, at the cost of one model's weights in VRAM per fold. No
  effect for a single fold.

### Python API

```python
import torch
from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming

predictor = nnUNetPredictorStreaming(
    tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
    perform_everything_on_device=True, device=torch.device('cuda', 0),
    verbose=False, verbose_preprocessing=False, allow_tqdm=True)
predictor.initialize_from_trained_model_folder(MODEL_FOLDER, use_folds=(0, 1))
predictor.predict_from_files_streaming(INPUT_FOLDER, OUTPUT_FOLDER)
```

`nnUNetPredictorStreaming(..., feature_ring=False)` keeps the logit buffer; `ring_dtype=torch.float32` on
`predict_from_files_streaming` / `predict_single_npy_array_streaming` is `--fp32_ring`. With the feature ring the
predictor's network returns features, so the logit-returning `nnUNetPredictor` methods
(`predict_sliding_window_return_logits`, `predict_logits_from_preprocessed_data`) raise until `restore_head()` is
called.

### Using it from another tool

A predictor is meant to be kept: initialize it once and call its `predict_*` methods as often as needed. The weights of
a single model stay on the GPU between calls, and the worker pools of batch prediction are started once and reused.
`close()` shuts the pools down; the predictor also works as a context manager.

**Images that are already in memory.** The NIfTI readers read nibabel / SimpleITK image objects as well as files
(`NibabelIO(WithReorient).read_nibabel_images`, `SimpleITKIO(WithReorient).read_sitk_images`) and return exactly what
`read_images` returns for the same images stored as files. `seg_to_nibabel` / `seg_to_sitk` return the image that
`write_seg` would write. Use the reader class of the model's plans (`plans.json`, `image_reader_writer`):

```python
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

io = NibabelIOWithReorient()
image, properties = io.read_nibabel_images([nib_image])       # nib_image: a nibabel.Nifti1Image
segmentation = predictor.predict_single_npy_array_streaming(image, properties)
seg_image = io.seg_to_nibabel(segmentation, properties)        # a nibabel image in nib_image's geometry
```

A single case runs entirely in the calling process: it is loaded and preprocessed in a background thread while the GPU
is prepared, so no worker processes are started.

**Batches with your own input and output handling.** `predict_cases_streaming` overlaps loading and preprocessing,
the GPU pass and the export like `nnUNetv2_predict_streaming` does, with two functions you provide:

- `load_fn(case) -> (image, properties)` runs in a preprocessing worker. `image` is `(c, x, y, z)` in the axis
  convention of the model's reader, `properties` needs at least `'spacing'`; extra keys are passed through to
  `export_fn`. Without `load_fn`, each case is a list of image files read with the plans' reader.
- `export_fn(segmentation, properties, output, **export_kwargs)` runs in an export worker and gets the finished label
  map in the input image's geometry. Its return values are returned in the order of the cases. Without `export_fn`,
  cases with an output (a file name without its ending) are written with nnU-Net's writer and cases without one return
  their segmentation.

Both have to be module-level functions, because they are sent to spawned worker processes, and cases, outputs and
`export_kwargs` have to be picklable:

```python
import numpy as np
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


def load_case(path):
    return SimpleITKIO().read_images([path])


def export_case(segmentation, properties, output):
    np.save(output, segmentation)
    return output


with nnUNetPredictorStreaming(device=torch.device('cuda', 0), quiet=True) as predictor:
    predictor.initialize_from_trained_model_folder(MODEL_FOLDER, use_folds=('all',))
    results = predictor.predict_cases_streaming(['a.nii.gz', 'b.nii.gz'], load_fn=load_case,
                                                outputs=['a_seg.npy', 'b_seg.npy'], export_fn=export_case,
                                                on_error='skip')
```

`on_error='skip'` turns a case whose loading, prediction or export fails into a `CaseFailure`
(`nnunetv2.inference.streaming_pipeline`) in the results, and the batch carries on; the default `'raise'` aborts the
batch. `progress_callback(event, index, total)` reports `'preprocessed'`, `'predicted'`, `'exported'` and `'failed'`
per case, `last_case_timings` holds the stage timings of the last call, and `num_parts` / `part_id` split a batch across
processes (one per GPU). `predict_single_case_streaming` takes the same `load_fn` / `export_fn` for one case.
`initialize_from_folder_base_trainer` builds the network with the base `nnUNetTrainer` when the trainer class named in
the checkpoint is not installed (fine for trainer variants that do not change the architecture).

### Giving a model new label ids

A tool that numbers its labels its own way can renumber a trained model instead of mapping ids after every
prediction. `nnUNetv2_reorder_labels` permutes the output channels of the segmentation layers and rewrites
`dataset.json`, so every voxel gets the same label name as before:

```bash
nnUNetv2_reorder_labels -i MODEL_FOLDER -o NEW_MODEL_FOLDER -l labels.json
```

`labels.json` holds the new `{label name: id}` mapping (a `dataset.json` works too). The new folder is for inference
only (no optimizer state or training logs). Region-based models are not supported.

## Probability export

`--save_probabilities` (`save_probabilities=True` in the Python API) writes the predicted probabilities (softmax over
the classes, or the sigmoid of each region for region-based models) to `<case>.b2nd` next to the segmentation, plus
`<case>.pkl` with the case's properties, as `nnUNetv2_predict` writes them. Each output slice is written as soon as it
is final, so the full probability volume never exists in memory, only on disk. Without `--save_probabilities` none of
this runs, and the segmentation is the same with and without it.

The file is a [blosc2](https://www.blosc.org/python-blosc2/) array of shape `(num_classes, *image_shape)`, in the axis
order of the model's reader, i.e. the layout of the `probabilities` array in the `.npz` files of `nnUNetv2_predict`.
Outside the region nnU-Net crops the image to, it holds what `nnUNetv2_predict` exports there (background 1; 0 for
region models). Values are stored as `uint16`, `round(p * 65535)`. The error is at most 7.6e-6, far below the
numerical noise of the prediction itself (half-precision network outputs). Rounding to multiples of 1/65535 turns the
tiny probabilities every class has far away from its structure into exact zeros, so the files are small. (Thresholding
stored region probabilities at 0.5 can flip voxels whose probability lies within 7.6e-6 of 0.5; the exported
segmentation is computed before rounding.)

### Reading the probabilities

The file is read lazily: slicing it decompresses only what you ask for, so a pipeline never has to load the whole
volume. With nnU-Net's helpers:

```python
from nnunetv2.inference.streaming_probabilities import load_streamed_probabilities

class_5 = load_streamed_probabilities('OUTPUT_FOLDER/case.b2nd', channels=5)  # one class, float32 probabilities
probs = load_streamed_probabilities('OUTPUT_FOLDER/case.b2nd')                # all classes, float32 (C, *image)
```

Without nnU-Net, blosc2 alone reads them:

```python
import blosc2
import numpy as np

arr = blosc2.open('OUTPUT_FOLDER/case.b2nd', mode='r', mmap_mode='r')  # leave out mmap_mode on Windows
scale = arr.schunk.vlmeta['nnunet']['scale']                            # 65535 for uint16 files, 1 for --float32
class_5 = arr[5].astype(np.float32) / scale                             # decompresses only that class
```

- **Channels.** For softmax models channel `c` is label `c`, and the channels sum to 1. For region-based models channel
  `c` is region `c` in the order of `label_manager.foreground_regions` (the regions of `dataset.json`); these are
  sigmoids and do not sum to 1. `arr.schunk.vlmeta['nnunet']['nonlinearity']` says which.
- **Axis order.** The reader's, as in the `.npz` of `nnUNetv2_predict`: voxel `[c, i, j, k]` lines up with the
  segmentation array as nnU-Net's reader returns it (for `SimpleITKIO`, the order of `sitk.GetArrayFromImage`).
- **Memory.** The whole volume as float32 needs `num_classes × voxels × 4` bytes, e.g. ~18 GB for 117 classes on a CT
  of 38 million voxels. Otherwise go class by class (`arr[c]`) or band by band (`arr[:, i0:i1]`).
- **Geometry.** It is in `<case>.pkl`. nnU-Net's `write_seg` casts to `uint8`, so to write a probability map as an
  image, set the geometry yourself. For `SimpleITKIO` (other readers keep it under other keys, e.g. `nibabel_stuff`):

```python
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import load_pickle

geometry = load_pickle('OUTPUT_FOLDER/case.pkl')['sitk_stuff']
img = sitk.GetImageFromArray(class_5)
img.SetSpacing(geometry['spacing'])
img.SetOrigin(geometry['origin'])
img.SetDirection(geometry['direction'])
sitk.WriteImage(img, 'OUTPUT_FOLDER/case_class_5.nii.gz')
```

### What it costs

Measured on an RTX 4090 with a Ryzen 9 7950X and a local NVMe SSD (one fold, no TTA): for 117 classes on
whole-body CTs of 38-46 million voxels the prediction took 0.3-0.6 s longer (3.4-4.1 s instead of 2.8-3.7 s) and the
probability files were 46-67 MB (9-11 GB as uncompressed `uint16`); for 125 classes on 512×512×~380 CTs it took 1-2 s
longer (6.3-7.4 s instead of 5.3-6.1 s), for 57-65 MB files (25 GB uncompressed). Peak VRAM grows by 0.15-0.75 GiB,
mostly two buffers of 4 output slices of all classes. The files are written by a background thread with 8 blosc2
threads (`nnUNetPredictorStreaming.probability_compression_threads`; `probability_planes_per_chunk` sets the slices per
buffer). `nnUNetv2_ensemble` does not read `.b2nd` files.

### Lossless float32 probabilities (`--float32`)

For those who need them, `--float32` stores the probabilities themselves (`predictor.probability_dtype = 'float32'`
in the Python API). It comes with a warning: floats keep the tiny probabilities, and their mantissas are noise, so
float32 compresses only 1.3x. On two of the CTs above that meant 14 GB (117 classes) and 36 GB (125 classes) per image
instead of 46 and 57 MB, and a prediction 2.5-4x slower (7-11 s instead of 2.7 s, 18-21 s instead of 6 s), with twice
the GPU buffer memory. There is no float16 option: its error near 1 is up to 2.4e-4 (`uint16`: 7.6e-6), and it
compresses only about 9x, so it would be less precise than `uint16` and still produce large files.

## Streaming final validation in training

The final validation of `nnUNetv2_train` (and `--val`) exports its predictions the way `nnUNetv2_predict` does, so with
many classes on large images it runs out of RAM. Trainers with `StreamingValidationMixin`
(`nnunetv2/training/nnUNetTrainer/variants/validation/nnUNetTrainer_streamingValidation.py`) validate with the streaming
predictor instead. Training is unchanged, resampling follows the plans as described above, and the training log says
which resampling and which slab buffer were used. 3d configurations only, no cascades, no `--npz`; other
configurations are refused when training starts.

- `nnUNetTrainer_SV`: `nnUNetTrainer` with streaming validation.
- `nnUNetTrainer_2000epochs_NoMirroring_SV`: 2000 epochs without mirroring.
- `nnUNetTrainer_DASegOrd0_NoMirroring_SV` and its `_2000epochs_SV` / `_5000epochs_SV` versions: a recipe for datasets
  with many classes. The spatial augmentation resamples the segmentation with nearest neighbour instead of
  interpolating one channel per class (which makes the augmentation CPU bound at 100+ classes), there is no
  mirroring, and the loss is chunked over the batch once it would exceed PyTorch's int32 indexing limit
  (`ChunkedLossMixin`; below that limit it is exactly the stock loss).

To add streaming validation to your own trainer, mix it in first:
`class MyTrainer_SV(StreamingValidationMixin, MyTrainer)`.

## Notes on memory and performance

- Peak VRAM is bounded by one slab of the internal prediction (≈ `channels × (patch_size_along_slab + 1) ×
  cross_section` in float16) plus one network forward and the preprocessed input volume (4 bytes per voxel and channel),
  instead of the full `num_classes × image` volume.
- The weights of a single model are uploaded to the GPU once per run, and the GPU memory cache is released at the end
  of each `predict_*` call, not after every case.
- **Feature ring (3d, on by default).** The network's last layer is a 1×1×1 convolution, and everything between it and
  the argmax (or the region threshold) is linear (test-time mirroring, Gaussian weighting, the sum over overlapping
  patches and folds, linear resampling). So the slab buffer holds the *input* of that layer — the decoder's last
  features, 32 channels for the standard U-Nets — instead of one channel per class, and the layer is applied once per
  output slice. The buffer does not grow with the number of classes (≈3.7× smaller at 120 classes). The layer is
  detached from the network before it is compiled, so this works with `--compile on`. Fold ensembles keep one block
  of features per fold, so they use it only while `folds × 32` is below the number of classes (e.g. up to 3 folds at
  117 classes); otherwise, for 2d configurations and for architectures whose output layer is not a known 1×1
  convolution, the buffer holds the logits. The runtime notice says which buffer is used and why. The segmentation is
  the same up to floating-point rounding (the sums run in a different order); `--logit_ring` turns it off.
- On CUDA with Triton, accumulating patches into that slab and the per-plane resample + argmax (region threshold for
  region-based models) run as fused GPU kernels that reproduce the plain torch computation bit for bit (checked once
  per process; everything falls back to the torch computation without Triton, e.g. on Windows, with identical
  results).
- Standalone streaming inference runs with `cudnn.benchmark` off: its autotuning costs time and a one-off VRAM peak
  in every new process for a forward pass that is only marginally faster.
- Throughput is comparable to `nnUNetv2_predict` (often slightly faster, since resampling runs on the GPU). Fold
  ensembling scales linearly with the number of folds.
- If even one slab does not fit, reduce VRAM by predicting on CPU (`-device cpu`) or, for very large 3d volumes,
  tiling a second axis is a possible future extension (not currently implemented).
