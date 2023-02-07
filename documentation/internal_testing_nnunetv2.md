This is an inofficial and rudimentary guide on how to work with nnU-Net v2. It assumes you are already familiar with 
nnU-Net.

Proper documentation will follow.

# Installation
nnUNetv2 depends on acvl_utils which is located on the Helmholtz GitLab. You need to 
[log in](https://gitlab.hzdr.de/users/sign_in?redirect_to_referer=yes) there once with your AD credentials, then 
contact me to get access!

Clone [acvl_utils](https://gitlab.hzdr.de/hi-dkfz/applied-computer-vision-lab/tools/acvl_utils) and then install
with `pip install -e .`

Ah and the same applies to [dynamic-network-architectures](https://gitlab.hzdr.de/hi-dkfz/applied-computer-vision-lab/tools/dynamic-network-architectures).

Then checkout the nnunet repository from phabricator
- http: https://phabricator.mitk.org/source/nnunet.git
- ssh (recommended): ssh://git@phabricator.mitk.org:2222/source/nnunet.git

Checkout the `nnunet_remake` branch. Pull and install with `pip install -e .`

IMPORTANT: Some datasets can be really slow if files from the training dataset are being opened over and over again. 
This only affects datasets with very high batch size, such as a 2d training of Hippocampus. Here, ~10k files are 
opened and closed per second. No, this is not a nnU-Net-specific problem.
You can circumvent this by setting `export nnUNet_keep_files_open=True` in your .bashrc. 
BUT you need to have permission to open sufficient files!
`ulimit -n` gives your current limit. It should not be 1024. 65535 works for me. See here how to change these limits:
[Link](https://kupczynski.info/posts/ubuntu-18-10-ulimits/) (works for Ubuntu 18, google for your Ubuntu version!).

Keeping file handles open does not affect 99% of trainings, so it's optional.

# Set up paths
Paths are called differently now!
Modify and add the following lines to your bashrc:

```bash
export nnUNet_preprocessed=/media/fabian/data/nnUNet_preprocessed
export nnUNet_raw=/media/fabian/data/nnUNet_raw
export nnUNet_results=/home/fabian/results/nnUNet_remake
```

# Dataset conversion
Datasets are a bit different now than in the old nnU-Net. The easiest way to use your old nnU-Net datasets is to run
```bash
nnUNetv2_convert_old_nnUNet_dataset INPUT_FOLDER OUTPUT_DATASET_NAME
```
Important: nnUNetv2 datasets are now called DatasetXXX_NAME. TaskXXX was a bad naming imposed by the MSD. So your 
dataset name should be Dataset004_Hippocampus, for example.

You can of course also build your own dataset, maybe even with a different file format (see 
[here](#list-of-major-new-features-no-particular-order)). No documentation on that yet, but sample datasets exist 
(see below for link)

When building your own dataset, use [generate_dataset_json](../nnunetv2/dataset_conversion/generate_dataset_json.py) 
to generate the dataset.json file. Read its documentation to see what the input arguments have to look like!
No headaches!

There are some sample datasets in `E132-Rohdaten/nnUNetv2` that you can use for inspiration. Notable examples are:
- `Dataset073_Fluo_C3DH_A549_SIM`: tif as input file format
- `Dataset120_RoadSegmentation`: 2d png as input file format
- `Dataset998_Hippocampus_ignore`: has an ignore label
- `Dataset999_Hippocampus_regions`: demonstrates how to use region-based training
- `Dataset997_sparseLiver`: training nnU-Net from scribbles. This was just an experiment, but a fun one. Toy dataset 
with just one training image (replicated 5 times).

If you set `export nnUNet_raw=[...]E132-Rohdaten/nnUNetv2` then you can just use the raw data located there. No need to copy.

# Standard nnU-Net v2 training procedure

Planning and preprocessing
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID
```
This will generate the dataset fingerprint (explicitly this time), plans and preprocessed data in 
`$nnUNet_preprocessed`. Everything is json now, so you can just open it in an editor, take a look and modify things 
(please only do the latter if you know what you are doing!)

Now train the desired configuration(s). Available: 2d, 3d_fullres, 3d_lowres, 3d_cascade fullres. The latter two of 
course only for datasets where they exist.
```bash
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD
```

After training is completed, find the best configuration
```bash
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID
```
You can specify the configurations you want to consider via `-c CONFIG1 CONFIG2 ...`. Or just leave at default.

Once this is completed it will tell you EXACTLY what commands you need to run for postprocessing and inference. 
Make sure to read the output!

In case you can't read:

Determine postprocessing:
```bash
nnUNetv2_determine_postprocessing
```
Apply postprocessing:
```bash
nnUNetv2_apply_postprocessing
```
Predict:
```bash
nnUNetv2_predict
```
Ensemble:
```bash
nnUNetv2_ensemble
```

IMPORTANT: ALL nnUNetv2* commands have a `-h` option. Use it!

# List of major new features (no particular order)
- No longer explicitly crops the raw data during planning and preprocessing. Saving some space
- Explicit separation of fingerprint extraction, experiment planning and preprocessing. See `nnUNetv2_extract_fingerprint`, `nnUNetv2_plan_experiment` and `nnUNetv2_preprocess` 
- Region based training (as in BraTS) is now natively supported
- We now support an ignore label. Name one of your labels 'ignore' to unlock this achievement
- Multiple input and output file formats are now supported:
  - Anything SimpleITK and nibabel can read/write
  - tif (requires separate .json file for each image/segmentation specifying spacing. Tif is a nightmare.)
  - png, bmp etc (lossless compression only, so no jpg (fck that shit))
  - (planned, Karol): HDF5, Zarr
  - (we still need the _0000 channel identifiers. Sorry.)
- you can now specify certain configurations (spacing, normalization, preprocessor) in the plans and then just 
preprocess them (no need to create custom experiment planners for everything). see `nnUNetv2_preprocess`
- you can swap out resampling functions for data, segmentation and probabilities (see plans file)
- cascade stage linking is now part of the plans and can be modified
- one lowres configuration can be linked to multiple fullres configurations
- (planned) Native integration of DDP training for single-machine, multi-GPU training
- substantial speed-up of cascaded data augmentation (it's still slow though!)
- much improved code documentation (in-line comments) and readability
- pedantic use of type hinting, even if it's ridiculous sometimes. You will see in the code.
- really fantastic code lines of death with nested list comprehensions and if statements. It's a joy to read those.
- much improved speed of postprocessing (and determination thereof)
- postprocessing that supports region-based training (not as simple as you may think)
- LabelManager to take care of all you label-related needs
- pytorch lightning-like trainer organization. NOT pytorch lightning though. F that shit.
- some more things that I probably forgot already

# Known bugs
- never save a dataset.json with sort_keys=True when using region-based training!