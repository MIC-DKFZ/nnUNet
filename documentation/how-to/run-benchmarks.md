# Run Benchmarks

This guide covers the built-in nnU-Net benchmark trainers.

## What the benchmark measures

nnU-Net provides two benchmark trainers that run for 5 epochs:

- `nnUNetTrainerBenchmark_5epochs`: full pipeline benchmark including data loading and augmentation
- `nnUNetTrainerBenchmark_5epochs_noDataLoading`: GPU-focused benchmark without data loading

Both write `benchmark_result.json` into the training output folder.

## Typical workflow

1. Prepare and preprocess a benchmark dataset
2. Run one or both benchmark trainers
3. Compare your epoch times against the published spreadsheet

## Benchmark commands

```bash
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainerBenchmark_5epochs
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainerBenchmark_5epochs_noDataLoading
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs_noDataLoading
```

Run only one benchmark training per GPU at a time.

## Where to look

Example output location:

```text
nnUNet_results/DatasetXXX_Name/nnUNetTrainerBenchmark_5epochs__nnUNetPlans__3d_fullres/fold_0/benchmark_result.json
```

## How to interpret results

- If both benchmark variants are slow, suspect GPU, PyTorch, CUDA, or system-level performance issues.
- If only the full-pipeline benchmark is slow, suspect data loading, storage throughput, or CPU bottlenecks.

Common bottlenecks:

- insufficient `nnUNet_n_proc_DA`
- slow `nnUNet_preprocessed` storage
- network filesystems with many small-file opens

## Published reference results

Published results are linked from the detailed benchmark page:

- [nnU-Netv2 benchmarks](../benchmarking.md)

## Related pages

- [Installation and setup](../getting-started/installation-and-setup.md)
- [Train models](train-models.md)

## Detailed legacy page

- [nnU-Netv2 benchmarks](../benchmarking.md)
