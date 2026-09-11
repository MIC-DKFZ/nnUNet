# Preprocessed data format

This page documents what nnU-Net writes into
`nnUNet_preprocessed/DatasetXXX_Name/<data_identifier>/` (for example `nnUNetPlans_3d_fullres/`),
and in particular how foreground sampling locations are stored.

## Files per configuration folder

| File | Contents |
| --- | --- |
| `<case>.b2nd` | Preprocessed image data, `(c, x, y, z)` float32, blosc2 |
| `<case>_seg.b2nd` | Preprocessed segmentation, `(1, x, y, z)`, blosc2 |
| `<case>.pkl` | Case properties: `spacing`, `shape_before_cropping`, `bbox_used_for_cropping`, `shape_after_cropping_and_before_resampling`, reader metadata, and `present_labels` |
| `fg_sampling/` | The foreground sampling location store, shared by all cases (see below) |

`present_labels` is the sorted list of label values that occur in the stored segmentation. It is
recorded during preprocessing, after `modify_seg_fn` and while the segmentation is still in memory,
so that building the sampling location store never has to scan the segmentations to find out which
labels a case contains. Migrating a dataset that predates this field costs nothing either: the
legacy `class_locations` entry already implies the same information (a class is empty there exactly
when none of its labels are present), and the migration pass reads it from the pkl it is replacing.

The store lives in its own subdirectory. Nothing that scans a configuration folder for cases sees
it, so case names are unconstrained.

## Foreground sampling locations

During training, nnU-Net oversamples patches centred on a foreground class
(`oversample_foreground_percent`, default 0.33). To do that, the dataloader needs **one** voxel
coordinate of a randomly chosen class that is present in the case.

nnU-Net used to keep those coordinates in each case's `.pkl` as
`properties['class_locations']`: a dict of class/region key to `(N, 4)` int64 coordinates. Getting
a single coordinate meant reading and unpickling the entire file. On datasets with many classes
that became the dominant cost of both dataloading and disk usage — on TotalSegmentator v2
(1228 cases, 117 classes) the `.pkl` files totalled **18.26 GB**, which is 28% of the whole
preprocessed folder and 76x more than every segmentation in it combined.

They now live in one compressed, partially readable store per configuration folder:

| File | Contents |
| --- | --- |
| `fg_sampling/locations.b2nd` | 1D `uint64` blosc2 array of flat (linear) voxel indices, sorted ascending within each (case, class) run |
| `fg_sampling/indptr.npy` | `int64`, length `n_cases + 1`. CSR row pointer |
| `fg_sampling/class_id.npy` | `uint16` (or `uint32`), length `nnz`. Index into the class key table |
| `fg_sampling/count.npy` | `uint32`, length `nnz`. Number of coordinates in that run |
| `fg_sampling/base.npy` | `int64`, length `n_cases`. Where each case's block starts |
| `fg_sampling/shape.npy` | `int64`, `(n_cases, 3)`. Spatial shape, used to unravel the linear indices |
| `fg_sampling/meta.json` | Store version, case order, class key table, sampling parameters |

Only non-empty (case, class) runs are stored, so the index is a genuine sparse (CSR) structure:
it grows with the number of classes *present per case*, not with the number of classes in the
dataset. For 100k cases with 10k potential classes and ~100 present per case, the index is 61 MB;
a dense table would be 4.0 GB.

Looking up one coordinate touches a handful of memory-mapped values and decompresses exactly one
blosc2 block:

```python
s, e     = indptr[case]; row = class_id[s:e]     # the classes present in this case
j        = searchsorted(row, class_id_of(key))
offset   = base[case] + count[s:s + j].sum()     # offsets are derived, not stored
lin      = locations[offset + randint(count[s + j])]
x, y, z  = unravel(lin, shape[case])
```

## Using it from code

```python
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

ds = infer_dataset_class(folder)(folder)
fg = ds.foreground_locations                  # store if present, legacy pkl fallback otherwise

fg.eligible_classes('case_042')               # classes/regions that have locations in this case
fg.count('case_042', 1)                       # how many locations class 1 has
fg.sample('case_042', 1)                      # one random (x, y, z), no channel axis
fg.all_locations('case_042', 1)               # all of them as (N, 3) — for inspection, not training
```

`load_case()` returns `(data, seg, seg_prev)` and deliberately does **not** read the `.pkl`. Use
`ds.get_properties(identifier)` when you need the case properties (validation and export do), and
`ds.get_shape(identifier)` when all you need is the spatial shape — that reads only the array
header.

Class keys are `int` for plain labels and `tuple` for regions, including the
`(-1, *all_labels)` key used when an ignore label is present (`LabelManager.annotated_classes_key`,
which both the extraction pass and the dataloader read it from). They round-trip through
`fg_sampling/meta.json` as tuples.

## Building and rebuilding the store

`nnUNetv2_plan_and_preprocess` and `nnUNetv2_preprocess` build it automatically. To (re)build it
yourself:

```bash
nnUNetv2_extract_sampling_locations -d DATASET_ID [-c 2d 3d_fullres] [-np 8] [--keep_existing]
```

This pass only reads the preprocessed **segmentations**, never the image data — 0.24 GB rather
than 45 GB on TotalSegmentator v2 — so it is cheap and can be re-run whenever you want, for
example after changing the sampling logic. On Hippocampus (260 cases) it takes about 2 seconds.

**Datasets preprocessed with an older nnU-Net keep working.** If no store is found, nnU-Net falls
back to reading `class_locations` from the per-case `.pkl` files and prints a one-time notice.
You do not need to preprocess again — running `nnUNetv2_extract_sampling_locations` migrates the
dataset in place. Re-running preprocessing also works and additionally shrinks the `.pkl` files.

## Why this format

Every parameter below was chosen by measurement on real TotalSegmentator v2 coordinates.

**Encoding.** A flat linear `uint64` index compresses to exactly the same size as `uint32` once
`SHUFFLE` + `BYTEDELTA` have run, so there is no reason to accept a voxel-count ceiling. Sorting
within each run is the single biggest lever, worth 4x. An `(N, 3) uint16` layout is 2.8x larger
and cannot use the fast single-element read path.

| filter pipeline (bytes per coordinate) | sorted | unsorted |
| --- | --- | --- |
| no filter | 1.69 | 3.33 |
| `SHUFFLE` | 1.26 | 3.01 |
| `SHUFFLE` + `BYTEDELTA` | **0.72** | 3.05 |

**Blocks.** In blosc2 the compression ratio is governed by the *block* size, while the chunk size
only matters through the number of blocks per chunk. `blocks=(512,)` sits at the cold-read minimum
and within a few percent of the best achievable size (measured on local NVMe):

| block (chunk 8192) | bytes/coord | warm read | cold read |
| --- | --- | --- | --- |
| 128 | 2.20 | 3.9 µs | 56.6 µs |
| **512** | **1.82** | **4.9 µs** | **50.8 µs** |
| 2048 | 1.67 | 9.3 µs | 52.0 µs |
| 8192 | 1.63 | 25.0 µs | 71.7 µs |

**Chunks.** `chunks=(32768,)`, i.e. 64 blocks per chunk. This is set by the *network filesystem*,
not by local behaviour. blosc2 pays a large fixed cost per chunk when writing to NFS (~16 ms,
against ~0.09 ms locally), so chunk count dominates write time there; and a cold random read costs
fewer metadata round-trips when there are fewer, larger chunks. Measured on an NFS-mounted cluster filesystem:

| chunk | blocks/chunk | write (7.8 MB) | warm read | cold read |
| --- | --- | --- | --- | --- |
| 8192 | 16 | 14.7 s | 7.9 µs | 131.0 µs |
| **32768** | **64** | **4.0 s** | **8.3 µs** | **71.6 µs** |
| 131072 | 256 | 1.5 s | 14.7 µs | 88.3 µs |
| 524288 | 1024 | 1.0 s | 37.6 µs | 119.7 µs |

32768 is 3.7x faster to write and 1.8x faster to read cold, for 5% on warm reads. Beyond it the
read cost climbs steeply as blocks-per-chunk grows.

**Reads** go through `schunk[i:i + 1]` rather than `NDArray[i]`, which is 4-5x faster because it
skips the NDArray slicing machinery. This is only equivalent to indexing the array because the
array is 1D — another reason for the linear encoding.

**The index is plain memory-mapped `.npy`, not blosc2.** A single draw touches four arrays with
short slices; storing them with blosc2 costs a block decompression on each and measured **130 µs
versus 4.5 µs**, to save 28 MB out of a multi-GB store.

**Everything is memory mapped** for reading, so dataloader workers share physical pages instead of
each holding a copy, and blosc2's chunk offset table is paged in lazily rather than read on open
(mmap is disabled on Windows, see issue #2723).

**The store is built in RAM and written once.** Growing a urlpath-backed blosc2 array incrementally
is free locally but pathological on a network filesystem, because every appended chunk rewrites
the frame's offset trailer. On NFS, writing 7.8 MB took 14.8 s that way versus 1.1 s when the
array is built in memory and serialised with a single `to_cframe()` + `write()` — 13x, growing with
dataset size. Preallocating the array, memory-mapped writing and larger flushes were all measured
and none of them helped; only removing the per-chunk filesystem traffic did. Stores too large to
hold in RAM (`max_in_memory_bytes`, 32 GiB compressed by default, so a ~64 GiB peak because
`to_cframe()` transiently holds a second copy) fall back to the incremental path. A 32 GiB store is
roughly 40 billion coordinates, so in practice that fallback is unreachable.

**Every blosc2 read is memory mapped** — the coordinate store, the preprocessed images and
segmentations in `load_case`, `get_shape`, and the segmentation reads in the extraction pass
(except on Windows, see issue #2723). Writing is the exception: the spill-to-disk fallback writes
without mmap, because a shared writable mapping over NFS has no reliable write-back ordering and
blosc2 needs the final file size up front, which is unknown while cases are still arriving.

## Measured effect

Hippocampus 2d (260 cases, 2 classes) — the small-dataset case, where the old format was already
cheap when warm:

| | before | after |
| --- | --- | --- |
| sampling metadata on disk | 27.59 MB in 260 files | 0.43 MB in 267 files |
| mean `.pkl` size | 106 KB | 523 bytes |
| one sampling draw, cold | 146.2 µs | 17.8 µs |
| one sampling draw, warm | 18.3 µs | 13.0 µs |

TotalSegmentator v2 3d_fullres (1228 cases, 117 classes) — the case this was built for:

| | before | after |
| --- | --- | --- |
| sampling metadata on disk | 18.26 GB in 1228 files | ~450 MB in 7 files |
| one sampling draw, cold | 6.73 ms | ~0.06 ms |
| one sampling draw, warm | 0.66 ms | ~0.03 ms |

Sampling behaviour itself is unchanged. The extraction pass calls the same
`DefaultPreprocessor._sample_foreground_locations` with the same parameters
(`min_num_samples=10000`, `min_percent_coverage=0.01`, `seed=1234`), and the store was verified to
reproduce the legacy coordinate sets exactly: 520 (case, class) pairs covering 856,754
coordinates on Hippocampus, zero mismatches.

## A note for custom preprocessors

Sampling locations are now extracted from the **stored** segmentation, i.e. after
`modify_seg_fn` has run, whereas `class_locations` used to be computed just before it. The default
`modify_seg_fn` is a no-op, so nothing changes for standard preprocessing. If you override it, be
aware that the locations now describe the segmentation training actually sees.
