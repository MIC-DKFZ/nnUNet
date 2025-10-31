# U2Net Planner

This directory contains the experiment planner for running U2Net within the nnUNet framework.

## Overview

The U2Net planner is responsible for configuring and preparing experiments using the U2Net architecture. It handles data preprocessing, experiment setup, and parameter selection specifically tailored for U2Net. You can find the U^2Net code in the [Dynamic Network Architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures) repository.

## Usage

- When preprocessing and planning, add the flag `-pl U2NetPlanner` to create plans for the U^2 Net. The plans file will be called "U2NetPlans.json".
- When training, use the flag `-p U2NetPlans`, to address the right plans. The results will be saved in a folder having the plans name.

For Dataset002_Heart (from the MSD), for example, this looks like this:

    nnUNet_results/
    ├── Dataset002_Heart
        │── nnUNetTrainer__U2NetPlans__2d
        │    ├── fold_0
        │    ├── fold_1
        │    ├── fold_2
        │    ├── fold_3
        │    ├── fold_4
        │    ├── dataset.json
        │    ├── dataset_fingerprint.json
        │    └── plans.json
        └── nnUNetTrainer__U2NetPlans__3d_fullres
             ├── fold_0
             ├── fold_1
             ├── fold_2
             ├── fold_3
             ├── fold_4
             ├── dataset.json
             ├── dataset_fingerprint.json
             └── plans.json

## Modifying the planner

Changing the planner can be useful for various purposes. \
The main parameters one can be interested in changing are the ones at the beginning of the init method.

```bash
# These are the maximum number of encoding stages for each configuration. The corresponding decoder stages will be n-1.
self.max_2d_stages = 6  
self.max_3d_stages = 5
```

```bash
# Here we set the depth of each of the RSU blocks in the encoding. The decoding stages will have the same deptbut in inverse order, starting from the second to last value. 

self.depth_per_stage = [7, 6, 5, 4, 4, 4]

# Please keep in mind you should change it consistently with the maximum stages.
# For optimal results, use decreasing values, or keep it the same. Increasing values usually bring to unwanted results.
```

``` bash
# These just overwrite the values in the Experiment Planner.
self.UNet_max_features_2d = 1024
self.UNet_max_features_3d = 512
```

## Tests

The U2Net planner includes a comprehensive test suite with 16 tests covering all major functionality and edge cases. The tests are located in `Tests/test_U2Net_planner.py` and can be run using pytest.

### Running Tests

```bash
pytest Tests/test_U2Net_planner.py -v
```

### Test Dependencies

The tests use mocking to avoid dependencies on actual nnUNet datasets and paths, making them fast and reliable. In this way no external data or GPU is required to run the tests.

## References

- [nnUNet Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [U2Net Paper](https://arxiv.org/abs/2005.09007)
- [Dynamic Network Architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)
