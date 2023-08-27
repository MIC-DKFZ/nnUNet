# TLDR Migration Guide from nnU-Net V1

- nnU-Net V2 can be installed simultaneously with V1. They won't get in each other's way
- The environment variables needed for V2 have slightly different names. Read [this](setting_up_paths.md). 
- nnU-Net V2 datasets are called DatasetXXX_NAME. Not Task.
- Datasets have the same structure (imagesTr, labelsTr, dataset.json) but we now support more 
[file types](dataset_format.md#supported-file-formats). The dataset.json is simplified. Use `generate_dataset_json` 
from nnunetv2.dataset_conversion.generate_dataset_json.py. 
- Careful: labels are now no longer declared as value:name but name:value. This has to do with [hierarchical labels](region_based_training.md). 
- nnU-Net v2 commands start with `nnUNetv2...`. They work mostly (but not entirely) the same. Just use the `-h` option.
- You can transfer your V1 raw datasets to V2 with `nnUNetv2_convert_old_nnUNet_dataset`. You cannot transfer trained 
models. Continue to use the old nnU-Net Version for making inference with those.
- These are the commands you are most likely to be using (in that order)
  - `nnUNetv2_plan_and_preprocess`. Example: `nnUNetv2_plan_and_preprocess -d 2`
  - `nnUNetv2_train`. Example: `nnUNetv2_train 2 3d_fullres 0`
  - `nnUNetv2_find_best_configuration`. Example: `nnUNetv2_find_best_configuration 2 -c 2d 3d_fullres`. This command
    will now create a `inference_instructions.txt` file in your `nnUNet_preprocessed/DatasetXXX_NAME/` folder which
    tells you exactly how to do inference.
  - `nnUNetv2_predict`. Example: `nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -c 3d_fullres -d 2`
  - `nnUNetv2_apply_postprocessing` (see inference_instructions.txt)
