Authors: \
Yannick Kirchhoff, Maximilian Rouven Rokuss, Benjamin Hamm, Ashis Ravindran, Constantin Ulrich, Klaus Maier-Hein<sup>&#8224;</sup>, Fabian Isensee<sup>&#8224;</sup>

&#8224;: equal contribution

# Introduction

This document describes our contribution to [Task 1 of the FLARE24 Challenge](https://www.codabench.org/competitions/2319/).
Our model is basically a default nnU-Net trained with larger batch size of 4 and 8, respectively. We submitted the batch size 8 model and an ensemble of the batch size 4 and batch size 8 models to the final test set.

# Experiment Planning and Preprocessing

Bring the downloaded data into the [nnU-Net format](../../../nnUNet/documentation/dataset_format.md) and add the dataset.json file as given here:

```json
{
    "name": "Dataset301_FLARE24Task1_labeled",
    "description": "Pan Cancer Segmentation",
    "labels": {
        "background": 0,
        "lesion": 1
    },
    "file_ending": ".nii.gz",
    "channel_names": {
        "0": "CT"
    },
    "numTraining": 5000
}
```

Afterwards you can run the default nnU-Net planning and preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 301 -c 3d_fullres
```

## Edit the plans files

In the generated `nnUNetPlans.json` file add the following configurations

```json
        "3d_fullres_bs4": {
            "inherits_from": "3d_fullres",
            "batch_size": 4
        },
        "3d_fullres_bs8": {
            "inherits_from": "3d_fullres",
            "batch_size": 8
        },
        "3d_fullres_bs4u8": {
            "inherits_from": "3d_fullres",
            "batch_size": 48
        }
```

Note, the last one is only used for the ensemble model during inference!

# Model training

Run the following commands to train the models with batch size 4 and 8. The large batch size helps stabilize the training despite the partial labels present in the dataset as well as handling the large number of scans in the dataset. We therefore keep the number of epochs at 1000.

```bash
nnUNetv2_train 301 3d_fullres_bs4 all

nnUNetv2_train 301 3d_fullres_bs8 all
```

# Inference

Our inference is optimized for efficient single scan prediction. For best performance, we strongly recommend running inference using the default `nnUNetv2_predict` command!

In order to run inference with the ensemble model you need to create a folder called `nnUNetTrainer__nnUNetPlans__3d_fullres_bs4u8` in the results folder and copy the `dataset.json`, `dataset_fingerprint.json` and `plans.json` from one of the other results folder as well as the `fold_all` from both trainings as `fold_0` and `fold_1`, respectively, into this new folder. This allows for easy ensembling of both models.

To run inference simply run the following commands with `folds` set to `all` for single model inference or `0 1` for the ensemble. `model_folder` is the folder containing the training results, i.e. for example `nnUNetTrainer__nnUNetPlans__3d_fullres_bs8`.

```bash
python inference_flare_task1.py -i input_folder -o output_folder -m model_folder -f folds
```
