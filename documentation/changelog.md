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