# CLI Command Catalog
There are 21 commands for nnUNetv2.

For basic use, there's only three commands that are needed:
- `nnUNetv2_plan_and_preprocess`
- `nnUNetv2_train`
- `nnUNetv2_predict`

# Available Commands
- [CLI Command Catalog](#cli-command-catalog)
- [Available Commands](#available-commands)
  - [Command: `nnUNetv2_extract_fingerprint`](#command-nnunetv2_extract_fingerprint)
    - [Arugments](#arugments)
  - [Command: `nnUNetv2_plan_experiment`](#command-nnunetv2_plan_experiment)
    - [Arugments](#arugments-1)
  - [Command: `nnUNetv2_preprocess`](#command-nnunetv2_preprocess)
    - [Arugments](#arugments-2)
  - [Command: `nnUNetv2_plan_and_preprocess`](#command-nnunetv2_plan_and_preprocess)
    - [Arugments](#arugments-3)
  - [Command: `nnUNetv2_train`](#command-nnunetv2_train)
    - [Arugments](#arugments-4)
  - [Command: `nnUNetv2_find_best_configuration`](#command-nnunetv2_find_best_configuration)
    - [Arugments](#arugments-5)
  - [Command: `nnUNetv2_accumulate_crossval_results`](#command-nnunetv2_accumulate_crossval_results)
    - [Arugments](#arugments-6)
  - [Command: `nnUNetv2_predict`](#command-nnunetv2_predict)
    - [Arugments](#arugments-7)
  - [Command: `nnUNetv2_predict_from_modelfolder`](#command-nnunetv2_predict_from_modelfolder)
    - [Arugments](#arugments-8)
  - [Command: `nnUNetv2_convert_old_nnUNet_dataset`](#command-nnunetv2_convert_old_nnunet_dataset)
    - [Arugments](#arugments-9)
  - [Command: `nnUNetv2_apply_postprocessing`](#command-nnunetv2_apply_postprocessing)
    - [Arugments](#arugments-10)
  - [Command: `nnUNetv2_determine_postprocessing`](#command-nnunetv2_determine_postprocessing)
    - [Arugments](#arugments-11)
  - [Command: `nnUNetv2_ensemble`](#command-nnunetv2_ensemble)
    - [Arugments](#arugments-12)
  - [Command: `nnUNetv2_plot_overlay_pngs`](#command-nnunetv2_plot_overlay_pngs)
    - [Arugments](#arugments-13)
  - [Command: `nnUNetv2_install_pretrained_model_from_zip`](#command-nnunetv2_install_pretrained_model_from_zip)
    - [Arugments](#arugments-14)
  - [Command: `nnUNetv2_export_model_to_zip`](#command-nnunetv2_export_model_to_zip)
    - [Arugments](#arugments-15)
  - [Command: `nnUNetv2_download_pretrained_model_by_url`](#command-nnunetv2_download_pretrained_model_by_url)
    - [Arugments](#arugments-16)
  - [Command: `nnUNetv2_move_plans_between_datasets`](#command-nnunetv2_move_plans_between_datasets)
    - [Arugments](#arugments-17)
  - [Command: `nnUNetv2_evaluate_folder`](#command-nnunetv2_evaluate_folder)
    - [Arugments](#arugments-18)
  - [Command: `nnUNetv2_evaluate_simple`](#command-nnunetv2_evaluate_simple)
    - [Arugments](#arugments-19)
  - [Command: `nnUNetv2_convert_MSD_dataset`](#command-nnunetv2_convert_msd_dataset)
    - [Arugments](#arugments-20)

---

## Command: `nnUNetv2_extract_fingerprint`

### Arugments
- `-d`
  - **Type**: `int`
  - **Required**: Yes
  - **Help**: [REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment planning, and preprocessing for these datasets. Can of course also be just one dataset.

- `-fpe`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `DatasetFingerprintExtractor`
  - **Help**: [OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is 'DatasetFingerprintExtractor'.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: [OPTIONAL] Number of processes used for fingerprint extraction. Default: `default_num_processes`.

- `--verify_dataset_integrity`
  - **Type**: `flag`
  - **Required**: No
  - **Default**: `False`
  - **Help**: [RECOMMENDED] Set this flag to check the dataset integrity. This is useful and should be done once for each dataset!

- `--clean`
  - **Type**: `flag`
  - **Required**: No
  - **Default**: `False`
  - **Help**: [OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a fingerprint already exists, the fingerprint extractor will not run.

- `--verbose`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! Recommended for cluster environments.

---

## Command: `nnUNetv2_plan_experiment`

### Arugments
- `-d`
  - **Type**: `int`
  - **Required**: Yes
  - **Help**: [REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment planning, and preprocessing for these datasets. Can of course also be just one dataset.

- `-pl`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `ExperimentPlanner`
  - **Help**: [OPTIONAL] Name of the Experiment Planner class that should be used. Default is 'ExperimentPlanner'. Note: There is no longer a distinction between 2d and 3d planner. It's an all-in-one solution now.

- `-gpu_memory_target`
  - **Type**: `float`
  - **Required**: No
  - **Default**: `None`
  - **Help**: [OPTIONAL] DANGER ZONE! Sets a custom GPU memory target (in GB). Default: None (Planner class default is used). Changing this will affect patch and batch size and will definitely affect your model's performance! Only use this if you really know what you are doing.

- `-preprocessor_name`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `DefaultPreprocessor`
  - **Help**: [OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in nnunetv2.preprocessing. Default: 'DefaultPreprocessor'. Changing this may affect your model's performance!

- `-overwrite_target_spacing`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `None`
  - **Help**: [OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and 3d_cascade_fullres configurations. Default: None [no changes]. Changing this will affect image size and potentially patch and batch size. This will definitely affect your model's performance! New target spacing must be a list of three numbers.

- `-overwrite_plans_name`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: [OPTIONAL] DANGER ZONE! If you used -gpu_memory_target, -preprocessor_name, or -overwrite_target_spacing, it is best practice to use -overwrite_plans_name to generate a differently named plans file such that the nnUNet default plans are not overwritten. You will then need to specify your custom plans file with -p whenever running other nnUNet commands (training, inference, etc).

---

## Command: `nnUNetv2_preprocess`

### Arugments
- `-d`
  - **Type**: `int`
  - **Required**: Yes
  - **Help**: [REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment planning, and preprocessing for these datasets. Can of course also be just one dataset.

- `-plans_name`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetPlans`
  - **Help**: [OPTIONAL] Specify a custom plans file that you may have generated.

- `-c`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `["2d", "3d_fullres", "3d_lowres"]`
  - **Help**: [OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3d_fullres 3d_lowres. Configurations that do not exist for some datasets will be skipped.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `None`
  - **Help**: [OPTIONAL] Define the number of processes to be used. If it's a list of numbers, it must match the number of configurations. More processes are faster but require more RAM. Default: 8 processes for 2d, 4 for 3d_fullres, 8 for 3d_lowres.

- `--verbose`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! Recommended for cluster environments.

---

## Command: `nnUNetv2_plan_and_preprocess`

### Arugments
- `-d`
  - **Type**: `int`
  - **Required**: Yes
  - **Help**: [REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment planning, and preprocessing for these datasets. Can of course also be just one dataset.

- `-fpe`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `DatasetFingerprintExtractor`
  - **Help**: [OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is 'DatasetFingerprintExtractor'.

- `-npfp`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `8`
  - **Help**: [OPTIONAL] Number of processes used for fingerprint extraction. Default: 8.

- `--verify_dataset_integrity`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [RECOMMENDED] Set this flag to check the dataset integrity. Useful for ensuring correctness.

- `--no_pp`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [OPTIONAL] Run only fingerprint extraction and experiment planning (no preprocessing). Useful for debugging.

- `--clean`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [OPTIONAL] Overwrite existing fingerprints. Required if changes are made to the Dataset Fingerprint Extractor or the dataset.

- `-pl`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `ExperimentPlanner`
  - **Help**: [OPTIONAL] Name of the Experiment Planner class that should be used. Default is 'ExperimentPlanner'.

- `-gpu_memory_target`
  - **Type**: `float`
  - **Required**: No
  - **Help**: [OPTIONAL] DANGER ZONE! Set a custom GPU memory target (in GB). Affects patch and batch size, influencing performance. Only use if experienced.

- `-preprocessor_name`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `DefaultPreprocessor`
  - **Help**: [OPTIONAL] DANGER ZONE! Use a custom preprocessor class located in nnunetv2.preprocessing. May affect model performance!

- `-overwrite_target_spacing`
  - **Type**: `list`
  - **Required**: No
  - **Help**: [OPTIONAL] DANGER ZONE! Set custom target spacing for specific configurations. Use with caution.

- `-overwrite_plans_name`
  - **Type**: `str`
  - **Required**: No
  - **Help**: [OPTIONAL] Use a custom plans identifier. Recommended when using other options like `-gpu_memory_target`.

- `-c`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `["2d", "3d_fullres", "3d_lowres"]`
  - **Help**: [OPTIONAL] Configurations for preprocessing. Default: 2d 3d_fullres 3d_lowres. Configurations that do not exist will be skipped.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `None`
  - **Help**: [OPTIONAL] Define the number of processes to use. Must match the number of configurations if provided as a list.

- `--verbose`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Print detailed logs. Useful for debugging.

---

## Command: `nnUNetv2_train`

### Arugments
- `dataset_name_or_id`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Dataset name or ID to train with.

- `configuration`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Configuration that should be trained.

- `fold`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Fold of the 5-fold cross-validation. Should be an int between 0 and 4.

- `-tr`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetTrainer`
  - **Help**: [OPTIONAL] Use this flag to specify a custom trainer. Default: `nnUNetTrainer`.

- `-p`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetPlans`
  - **Help**: [OPTIONAL] Use this flag to specify a custom plans identifier. Default: `nnUNetPlans`.

- `-pretrained_weights`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: [OPTIONAL] Path to nnU-Net checkpoint file to be used as pretrained model. Will only be used when actually training. Beta. Use with caution.

- `-num_gpus`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `1`
  - **Help**: Specify the number of GPUs to use for training.

- `--use_compressed`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [OPTIONAL] If set, the training cases will not be decompressed. Reading compressed data is more CPU and RAM intensive and should only be used if you know what you are doing.

- `--npz`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [OPTIONAL] Save softmax predictions from final validation as `.npz` files (in addition to predicted segmentations). Needed for finding the best ensemble.

- `--c`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [OPTIONAL] Continue training from the latest checkpoint.

- `--val`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [OPTIONAL] Set this flag to only run the validation. Requires training to have finished.

- `--val_best`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [OPTIONAL] If set, validation will be performed with `checkpoint_best` instead of `checkpoint_final`. NOT COMPATIBLE with `--disable_checkpointing`.

- `--disable_checkpointing`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: [OPTIONAL] Disable checkpointing. Ideal for testing and preventing the hard drive from filling up with checkpoints.

- `-device`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `cuda`
  - **Help**: Specify the device for training. Options are `cuda` (GPU), `cpu` (CPU), or `mps` (Apple M1/M2). Do NOT use this to set which GPU ID! Use `CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...]` instead!

---

## Command: `nnUNetv2_find_best_configuration`

### Arugments
- `dataset_name_or_id`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Dataset Name or ID.

- `-p`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `["nnUNetPlans"]`
  - **Help**: List of plan identifiers. Default: `nnUNetPlans`.

- `-c`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]`
  - **Help**: List of configurations. Default: `['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']`.

- `-tr`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `["nnUNetTrainer"]`
  - **Help**: List of trainers. Default: `nnUNetTrainer`.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: Number of processes to use for ensembling, postprocessing, etc.

- `-f`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `(0, 1, 2, 3, 4)`
  - **Help**: Folds to use. Default: `0 1 2 3 4`.

- `--disable_ensembling`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Set this flag to disable ensembling.

- `--no_overwrite`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: If set, will not overwrite already ensembled files, potentially speeding up consecutive runs.

---

## Command: `nnUNetv2_accumulate_crossval_results`

### Arugments
- `dataset_name_or_id`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Dataset Name or ID.

- `-c`
  - **Type**: `str`
  - **Required**: Yes
  - **Default**: `3d_fullres`
  - **Help**: Configuration to use. Default: `3d_fullres`.

- `-o`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Output folder. If not specified, the output folder will be located in the trained model directory (named `crossval_results_folds_XXX`).

- `-f`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `(0, 1, 2, 3, 4)`
  - **Help**: Folds to use. Default: `0 1 2 3 4`.

- `-p`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetPlans`
  - **Help**: Plan identifier in which to search for the specified configuration. Default: `nnUNetPlans`.

- `-tr`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetTrainer`
  - **Help**: Trainer class. Default: `nnUNetTrainer`.

---

## Command: `nnUNetv2_predict`

### Arugments
- `-i`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Input folder. Ensure the correct channel numbering for your files (e.g., `_0000`). File endings must match the training dataset.

- `-o`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Output folder. If it does not exist, it will be created. Predicted segmentations will have the same name as their source images.

- `-m`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Folder containing the trained model. Must have subfolders `fold_X` for the different folds you trained.

- `-f`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `(0, 1, 2, 3, 4)`
  - **Help**: Specify the folds of the trained model to be used for prediction. Default: `(0, 1, 2, 3, 4)`.

- `-step_size`
  - **Type**: `float`
  - **Required**: No
  - **Default**: `0.5`
  - **Help**: Step size for sliding window prediction. Larger values are faster but less accurate. Cannot exceed 1. Default: `0.5`.

- `--disable_tta`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Disable test-time augmentation (mirroring). Faster but less accurate inference. Not recommended.

- `--verbose`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Enable verbose output.

- `--save_probabilities`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Export predicted class probabilities. Required for ensembling multiple configurations.

- `--continue_prediction`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Continue an aborted previous prediction (will not overwrite existing files).

- `-chk`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `checkpoint_final.pth`
  - **Help**: Name of the checkpoint to use. Default: `checkpoint_final.pth`.

- `-npp`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `3`
  - **Help**: Number of processes used for preprocessing. Beware of out-of-RAM issues. Default: `3`.

- `-nps`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `3`
  - **Help**: Number of processes used for segmentation export. Beware of out-of-RAM issues. Default: `3`.

- `-prev_stage_predictions`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Folder containing the predictions of the previous stage. Required for cascaded models.

- `-device`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `cuda`
  - **Help**: Set the device for inference. Options: `cuda` (GPU), `cpu` (CPU), `mps` (Apple M1/M2). Do not use this to set GPU ID.

- `--disable_progress_bar`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Disable the progress bar. Recommended for non-interactive HPC environments.

---

## Command: `nnUNetv2_predict_from_modelfolder`

### Arugments
- `-i`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Input folder. Ensure the correct channel numbering for your files (e.g., `_0000`). File endings must match the training dataset.

- `-o`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Output folder. If it does not exist, it will be created. Predicted segmentations will have the same name as their source images.

- `-d`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Dataset name or ID with which you would like to predict.

- `-p`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetPlans`
  - **Help**: Plans identifier. Specify the plans in which the desired configuration is located. Default: `nnUNetPlans`.

- `-tr`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetTrainer`
  - **Help**: nnU-Net trainer class used for training. Default: `nnUNetTrainer`.

- `-c`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: nnU-Net configuration to be used for prediction. Must be located in the plans specified with `-p`.

- `-f`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `(0, 1, 2, 3, 4)`
  - **Help**: Folds of the trained model to be used for prediction. Default: `(0, 1, 2, 3, 4)`.

- `-step_size`
  - **Type**: `float`
  - **Required**: No
  - **Default**: `0.5`
  - **Help**: Step size for sliding window prediction. Larger values are faster but less accurate. Cannot exceed 1. Default: `0.5`.

- `--disable_tta`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Disable test-time augmentation (mirroring). Faster but less accurate inference. Not recommended.

- `--verbose`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Enable verbose output.

- `--save_probabilities`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Export predicted class probabilities. Required for ensembling multiple configurations.

- `--continue_prediction`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Continue an aborted previous prediction (will not overwrite existing files).

- `-chk`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `checkpoint_final.pth`
  - **Help**: Name of the checkpoint to use. Default: `checkpoint_final.pth`.

- `-npp`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `3`
  - **Help**: Number of processes used for preprocessing. Beware of out-of-RAM issues. Default: `3`.

- `-nps`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `3`
  - **Help**: Number of processes used for segmentation export. Beware of out-of-RAM issues. Default: `3`.

- `-prev_stage_predictions`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Folder containing the predictions of the previous stage. Required for cascaded models.

- `-num_parts`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `1`
  - **Help**: Number of separate `nnUNetv2_predict` calls to be made. Default: `1` (predicts everything in one call).

- `-part_id`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `0`
  - **Help**: Identifier for this prediction part. Must be between `0` and `num_parts - 1`.

- `-device`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `cuda`
  - **Help**: Set the device for inference. Options: `cuda` (GPU), `cpu` (CPU), `mps` (Apple M1/M2). Do not use this to set GPU ID.

- `--disable_progress_bar`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Disable the progress bar. Recommended for non-interactive HPC environments.

---

## Command: `nnUNetv2_convert_old_nnUNet_dataset`

### Arugments
- `input_folder`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Path to the raw old nnUNet dataset folder containing `imagesTr`, `labelsTr`, etc. Provide the full path to the old task, not just the task name. nnU-Net V2 does not know where v1 tasks are located.

- `output_dataset_name`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: New dataset name following the `DatasetXXX_NAME` convention. This is the name, not the path.

---

## Command: `nnUNetv2_apply_postprocessing`

### Arugments
- `-i`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Input folder where the postprocessing should be applied.

- `-o`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Output folder where the postprocessed results will be saved.

- `-pp_pkl_file`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Path to the `postprocessing.pkl` file containing the postprocessing instructions.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: Number of processes to use. Default: `default_num_processes`.

- `-plans_json`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Plans file to use. If not specified, the tool will look for `plans.json` in the input folder (`input_folder/plans.json`).

- `-dataset_json`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Dataset file to use. If not specified, the tool will look for `dataset.json` in the input folder (`input_folder/dataset.json`).

---

## Command: `nnUNetv2_determine_postprocessing`

### Arugments
- `-i`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Input folder where the postprocessing files will be written.

- `-ref`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Folder containing the ground truth labels.

- `-plans_json`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Plans file to use. If not specified, the tool will look for `plans.json` in the input folder (`input_folder/plans.json`).

- `-dataset_json`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Dataset file to use. If not specified, the tool will look for `dataset.json` in the input folder (`input_folder/dataset.json`).

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: Number of processes to use. Default: `default_num_processes`.

- `--remove_postprocessed`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Set this flag if you do not want to keep the postprocessed files.

---

## Command: `nnUNetv2_ensemble`

### Arugments
- `-i`
  - **Type**: `list`
  - **Required**: Yes
  - **Help**: List of input folders to ensemble.

- `-o`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Output folder where the ensembled results will be saved.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: Number of processes to use for ensembling. Default: `default_num_processes`.

- `--save_npz`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Set this flag to store output probabilities in separate `.npz` files.

---

## Command: `nnUNetv2_plot_overlay_pngs`

### Arugments
- `-d`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Dataset name or ID.

- `-o`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Output folder where the PNG overlays will be saved.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: Number of processes used. Default: `default_num_processes`.

- `-channel_idx`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `0`
  - **Help**: Channel index used (e.g., `0 = _0000`). Default: `0`.

- `--use_raw`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: If set, raw data is used; otherwise, preprocessed data is used.

- `-p`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetPlans`
  - **Help**: Plans identifier. Only used if `--use_raw` is not set. Default: `nnUNetPlans`.

- `-c`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Configuration name. Only used if `--use_raw` is not set. Default: `None` (uses `3d_fullres` if available, else `2d`).

- `-overlay_intensity`
  - **Type**: `float`
  - **Required**: No
  - **Default**: `0.6`
  - **Help**: Overlay intensity. Higher values result in brighter/less transparent overlays.

---

## Command: `nnUNetv2_install_pretrained_model_from_zip`

### Arugments
- `zip`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Path to the zip file containing the pretrained model.

---

## Command: `nnUNetv2_export_model_to_zip`

### Arugments
- `-d`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Dataset name or ID.

- `-o`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Output file name for the exported zip file.

- `-c`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `("3d_lowres", "3d_fullres", "2d", "3d_cascade_fullres")`
  - **Help**: List of configuration names to include in the export.

- `-tr`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetTrainer`
  - **Help**: Trainer class used.

- `-p`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `nnUNetPlans`
  - **Help**: Plans identifier.

- `-f`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `(0, 1, 2, 3, 4)`
  - **Help**: List of fold IDs to include in the export.

- `-chk`
  - **Type**: `list`
  - **Required**: No
  - **Default**: `("checkpoint_final.pth",)`
  - **Help**: List of checkpoint names to export. Default: `checkpoint_final.pth`.

- `--not_strict`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Allow missing folds and/or configurations.

- `--exp_cv_preds`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Export the cross-validation predictions as well.

---

## Command: `nnUNetv2_download_pretrained_model_by_url`

### Arugments
- `url`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: URL of the pretrained model to download.

---

## Command: `nnUNetv2_move_plans_between_datasets`

### Arugments
- `-s`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Source dataset name or ID.

- `-t`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Target dataset name or ID.

- `-sp`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Source plans identifier. If your plans are named `nnUNetPlans.json`, then the identifier would be `nnUNetPlans`.

- `-tp`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Target plans identifier. If not provided, the source plans identifier will be kept. This is not recommended if the source identifier is a default nnU-Net identifier like `nnUNetPlans`.

---

## Command: `nnUNetv2_evaluate_folder`

### Arugments
- `gt_folder`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Folder containing ground truth segmentations.

- `pred_folder`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Folder containing predicted segmentations.

- `-djfile`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Path to the `dataset.json` file.

- `-pfile`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Path to the `plans.json` file.

- `-o`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `pred_folder/summary.json`
  - **Help**: Output file for the summary of the evaluation. If not provided, the default is `pred_folder/summary.json`.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: Number of processes to use. Default: `default_num_processes`.

- `--chill`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Do not crash if `pred_folder` does not have all files that are present in `gt_folder`.

---

## Command: `nnUNetv2_evaluate_simple`

### Arugments
- `gt_folder`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Folder containing ground truth segmentations.

- `pred_folder`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Folder containing predicted segmentations.

- `-l`
  - **Type**: `list`
  - **Required**: Yes
  - **Help**: List of labels to evaluate.

- `-il`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Label to ignore during evaluation.

- `-o`
  - **Type**: `str`
  - **Required**: No
  - **Default**: `pred_folder/summary.json`
  - **Help**: Output file for the summary of the evaluation. If not provided, the default is `pred_folder/summary.json`.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: Number of processes to use. Default: `default_num_processes`.

- `--chill`
  - **Type**: `flag`
  - **Required**: No
  - **Help**: Do not crash if `pred_folder` does not have all files that are present in `gt_folder`.

---

## Command: `nnUNetv2_convert_MSD_dataset`

### Arugments
- `-i`
  - **Type**: `str`
  - **Required**: Yes
  - **Help**: Path to the downloaded and extracted MSD dataset folder. **Cannot be an nnUNetv1 dataset**. Example: `/home/user/Downloads/Task05_Prostate`.

- `-overwrite_id`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `None`
  - **Help**: Overwrite the dataset ID. If not set, the ID of the MSD task (inferred from the folder name) is used. Only use this if you already have an equivalently numbered dataset.

- `-np`
  - **Type**: `int`
  - **Required**: No
  - **Default**: `default_num_processes`
  - **Help**: Number of processes to use. Default: `default_num_processes`.
