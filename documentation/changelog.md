# Changes that affect custom trainers

## Segmentation resampling: ties are broken differently, and the memory efficient branch is now the default

`resample_torch_simple` and `resample_torch_fornnunet` resample a segmentation by interpolating each
label's indicator and picking a winner per voxel. So do `resample_data_or_seg` and its two wrappers on
the default (non-torch) path, by way of batchgenerators' `resize_segmentation`; they take the same new
`seg_tiebreak` argument, with the same default. Whenever an axis is resampled by an even integer
factor, every boundary voxel is an exact 50/50 tie between two labels - in plane (512 -> 256) just as
much as out of plane. On the torch path those ties used to be resolved by `argmax`, which always returns
the first maximum and therefore always picked the **smallest** label, i.e. background for a normal
segmentation. That eroded foreground systematically and annihilated structures one voxel thick.

nnU-Net now requires **batchgenerators >= 0.25.4**, which is where `resize_segmentation` gained
`seg_tiebreak`, and **batchgeneratorsv2 >= 0.3.6**, whose `SpatialTransform` changes the augmented
segmentations every training sees in two ways:

- **The same tie-break**, `seg_tiebreak='nearest'`. nnU-Net passes `bg_style_seg_sampling=False`, whose ties
  used to go to the smallest label. Exact ties are rarer here than in preprocessing (about one voxel in a
  million under random rotation and scaling), but they are not zero.
- **Segmentations are padded with the padding label, then resampled**, so they end at the outer edge of the
  border pixels, like the image. A rotated or scaled patch that reaches past the loaded crop used to cut
  labels half a pixel short there. Under nnU-Net's settings (`border_mode_seg='constant'`,
  `padding_value_seg=-1`) this turns -1 (later background) into the real label at crop edges: 0.64 % of the
  voxels of a patch that is both rotated and scaled, none in a patch that stays inside the crop.

Custom trainers that build their own `SpatialTransform` get both changes. `seg_tiebreak='lowest'` restores
the old tie direction; the padding change has no switch. `bg_style_seg_sampling` now defaults to `False`
there, which gives the same output as `True` and is faster.

Two new defaults:

- **`seg_tiebreak='nearest'`** (new parameter). Undecided voxels now take the nearest neighbour label,
  which is the decision the separate-z path already made and is symmetric in the labels - provided it is
  one of the tied labels. Where three or more labels meet in 3D it can be a label that lost, and such a
  voxel keeps the smallest of the tied labels instead. `'lowest'` and
  `'highest'` keep the smallest or the largest label instead; all three are accepted on both paths, and an
  unknown value is rejected up front.
- **`memefficient_seg_resampling=True`**. The two seg implementations are now bit-identical, so this
  is purely a time/memory trade: the previous default scores all labels into one
  `n_labels x voxels x 2` byte buffer, this one holds a single running score. It is also the faster of
  the two now. Measured on CPU with `seg_tiebreak='nearest'`, upsampling 128x256x256 to 160x320x320:
  1.5x faster at 3 labels at the same peak memory, 2.8x faster at 20 labels (peak RSS 2.3 GB -> 0.9 GB),
  and 2.4x faster at 100 labels from 96x192x192 (7.3 GB -> 0.9 GB). Downsampling 200x512x512 to
  100x256x256 it is 1.2x faster at 3 labels and 2.3x at 20. It handles every interpolation mode
  `F.interpolate` does.

**This changes preprocessed segmentations**, and with them what a model is trained on. Exported
predictions of a single-stage configuration are not affected: they are resampled as probabilities and only
argmaxed afterwards. The cascade is affected at inference: the low resolution stage's prediction is
resampled to the full resolution grid as a segmentation, with this code, and fed to the next stage. Measured
over 100 segmentations from 18 datasets, roundtripping each through its target grid and back: mean
foreground Dice +0.010 at an even integer factor on the anisotropic axis (+0.012 at 4x, +0.005 in plane),
foreground volume ratio 0.912 -> 0.998, and +0.027 Dice on structures under 1000 voxels. Where no exact
ties occur - odd factors, and most real target spacings - the two are identical or within 0.0002 Dice.

Plans files do **not** pin `seg_tiebreak` and will therefore pick up the new tie-break. Plans from the
torch planners do pin `memefficient_seg_resampling`, so they keep the value they were created with; the
default planners' plans resample with `resample_data_or_seg_to_shape`, which has no such option. To get the
old results back, set `"seg_tiebreak"` in `resampling_fn_seg_kwargs` of your plans file:

- **torch planners**: `"lowest"`. Together with `"memefficient_seg_resampling": false` - what older plans
  files already say - this reproduces the old results bit for bit.
- **default planners** (`nnUNetPlans`, the ResEnc presets): `"highest"` is the closest, but nothing
  reproduces the old results exactly, because what they did was a bug. The old code compared each label
  against 0.5 rather than against the other labels, and left voxels that no label claimed at 0 (see
  below). Where no voxel went unclaimed, `"highest"` gives the old result at all but very rare voxels:
  floating-point rounding can let two labels both reach 0.5 with slightly different scores, and there the
  old code took the larger label while the argmax takes the higher score.

The `memefficient_seg_resampling` branch also had a bug worth knowing about if you ever enabled it: it
assigned labels with `interp > 0.5`, so a voxel no label claimed kept the zero the result was
initialized with. With labels `{1, 2}` that produced a 0 that does not occur in the input at all.

The default path had that same bug, in batchgenerators rather than here (fixed in 0.25.4). It bit harder
there: `resize_segmentation` defaults to spline order 3, and skimage's `clip=True` breaks the sum-to-one
property that the 0.5 threshold relies on, so on random four-label 2d data 6.2 % of output voxels had no
label at 0.5 or above. Ties among the labels that did pass 0.5 went to the **largest** label, not the
smallest, because the loop assigned in ascending label order and the last writer won - the opposite
direction from the bug above, and the reason the default path did not show the same foreground erosion.
The separate-z pass along the anisotropic axis did the same thing in nnU-Net itself whenever `order_z` was
not 0. All of these now take the per-voxel argmax with `seg_tiebreak`. The default plans use `order_z=0`
for segmentations, and that pass is unchanged.

`seg_tiebreak` is keyword-only everywhere, so it can be reordered later without breaking positional
callers.

`resample_torch_fornnunet` now honours its `mode` argument when it does not resample separately along z.
It used to drop it there and always interpolate linearly. No shipped planner sets `mode`, so this only
changes plans files that set it in their resampling kwargs.

## DDP: `do_split` must be called by all ranks

In DDP, `nnUNetTrainer.do_split` now lets only global rank 0 read or create `splits_final.json` and broadcasts the
splits to the other ranks (`dist.broadcast_object_list`). Previously every rank could create the file at the same
time, and a rank could read a file that another rank was still writing. Because of the broadcast, `do_split` is now a
collective: if a custom trainer calls it on some ranks only (for example inside `if self.global_rank == 0:`), the job
hangs. Call it on all ranks. Trainers that override `do_split` entirely are not affected.

## 2D TIFF predictions use PackBits; `.tiff` may select NaturalImage2DIO

`NaturalImage2DIO.write_seg` now writes `.tif` / `.tiff` predictions with PackBits via `tifffile` instead of uncompressed `skimage.io.imsave`. Pixel values and dtypes are unchanged (still lossless). PNG/BMP writes and `Tiff3DIO` (zlib) are unchanged. Downstream tools that digest the raw TIFF byte stream (not just the label map) will see a different container.

`NaturalImage2DIO` also claims `.tiff` and is listed before `Tiff3DIO`. With an example file, registry selection still falls through to `Tiff3DIO` for non-RGB 3D volumes. Calling `determine_reader_writer_from_file_ending('.tiff', None)` (no example file) now returns `NaturalImage2DIO` instead of `Tiff3DIO`; pass an example file when you need 3D disambiguation.

## Multi-node DDP: `local_rank` is no longer the rank you want for file writes

nnU-Net now supports DDP across several nodes (see [Multi-GPU training](multi_gpu_training.md)). To make that work,
`nnUNetTrainer` distinguishes three quantities where it previously only had `self.local_rank`:

| attribute | meaning |
| --- | --- |
| `self.local_rank` | index of this process **within its node**. This is the CUDA device index. |
| `self.global_rank` | index of this process **within the whole job**. Exactly one process in the job has 0. |
| `self.world_size` | number of processes in the job |
| `self.local_world_size` | number of processes on this node |

On a single node `local_rank == global_rank`, so nothing changes. Across nodes they differ, and **every check that
decides who writes to `nnUNet_results` must use `global_rank`** — there is one `local_rank == 0` per node, and on a
shared filesystem they would all write the same log file, checkpoints and progress plot. All in-tree trainers were
converted. If you maintain a custom trainer, replace `if self.local_rank == 0:` with `if self.global_rank == 0:`
wherever it guards output, and keep `local_rank` only where you mean the GPU.

`AllGatherGrad` (`nnunetv2.utilities.ddp_allgather`) is deprecated and will be removed in a future release.
nnU-Net's losses now use `AllReduceGrad` (`nnunetv2.utilities.ddp`), which computes the same forward and
backward while moving `world_size` times less data: `AllGatherGrad.apply(x).sum(0)` is exactly
`AllReduceGrad.apply(x)`. Importing `AllGatherGrad` still works but raises a `DeprecationWarning`.

# What is different in v2?

- We now support **hierarchical labels** (named regions in nnU-Net). For example, instead of training BraTS with the
'edema', 'necrosis' and 'enhancing tumor' labels you can directly train it on the target areas 'whole tumor',
'tumor core' and 'enhancing tumor'. See [here](region_based_training.md) for a detailed description + also have a look at the
[BraTS 2021 conversion script](../nnunetv2/dataset_conversion/Dataset137_BraTS21.py).
- Cross-platform support. Cuda, mps (Apple M1/M2) and of course CPU support! Simply select the device with
`-device` in `nnUNetv2_train` and `nnUNetv2_predict`.
- Unified trainer class: nnUNetTrainer. No messing around with cascaded trainer, DDP trainer, region-based trainer,
ignore trainer etc. All default functionality is in there!
- Supports more input/output data formats through ImageIO classes.
- I/O formats can be extended by implementing new Adapters based on `BaseReaderWriter`.
- The nnUNet_raw_cropped folder no longer exists -> saves disk space at no performance penalty. magic! (no jk the
saving of cropped npz files was really slow, so it's actually faster to crop on the fly).
- Preprocessed data and segmentation are stored in different files when unpacked. Seg is stored as int8 and thus
takes 1/4 of the disk space per pixel (and I/O throughput) as in v1.
- Native support for multi-GPU (DDP) TRAINING.
Multi-GPU INFERENCE should still be run with `CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] -num_parts Y -part_id X`.
There is no cross-GPU communication in inference, so it doesn't make sense to add additional complexity with DDP.
- All nnU-Net functionality is now also accessible via API. Check the corresponding entry point in `setup.py` to see
what functions you need to call.
- Dataset fingerprint is now explicitly created and saved in a json file (see nnUNet_preprocessed).

- Complete overhaul of plans files (read also [this](explanation_plans_files.md):
  - Plans are now .json and can be opened and read more easily
  - Configurations are explicitly named ("3d_fullres" , ...)
  - Configurations can inherit from each other to make manual experimentation easier
  - A ton of additional functionality is now included in and can be changed through the plans, for example normalization strategy, resampling etc.
  - Stages of the cascade are now explicitly listed in the plans. 3d_lowres has 'next_stage' (which can also be a
  list of configurations!). 3d_cascade_fullres has a 'previous_stage' entry. By manually editing plans files you can
  now connect anything you want, for example 2d with 3d_fullres or whatever. Be wild! (But don't create cycles!)
  - Multiple configurations can point to the same preprocessed data folder to save disk space. Careful! Only
  configurations that use the same spacing, resampling, normalization etc. should share a data source! By default,
  3d_fullres and 3d_cascade_fullres share the same data
  - Any number of configurations can be added to the plans (remember to give them a unique "data_identifier"!)

Folder structures are different and more user-friendly:
- nnUNet_preprocessed
  - By default, preprocessed data is now saved as: `nnUNet_preprocessed/DATASET_NAME/PLANS_IDENTIFIER_CONFIGURATION` to clearly link them to their corresponding plans and configuration
  - Name of the folder containing the preprocessed images can be adapted with the `data_identifier` key.
- nnUNet_results
  - Results are now sorted as follows: DATASET_NAME/TRAINERCLASS__PLANSIDENTIFIER__CONFIGURATION/FOLD

## What other changes are planned and not yet implemented?
- Integration into MONAI (together with our friends at Nvidia)
- New pretrained weights for a large number of datasets (coming very soon))


[//]: # (- nnU-Net now also natively supports an **ignore label**. Pixels with this label will not contribute to the loss. )

[//]: # (Use this to learn from sparsely annotated data, or excluding irrelevant areas from training. Read more [here]&#40;ignore_label.md&#41;.)
