Authors: \
Yannick Kirchhoff*, Ashis Ravindran*, Maximilian Rouven Rokuss, Benjamin Hamm, Constantin Ulrich, Klaus Maier-Hein<sup>&#8224;</sup>, Fabian Isensee<sup>&#8224;</sup>

*: equal contribution \
&#8224;: equal contribution

# Introduction

This document describes our contribution to [Task 2 of the FLARE24 Challenge](https://www.codabench.org/competitions/2320/).
Our model is basically a default nnU-Net with a custom low resolution setting and OpenVINO optimizations for faster CPU inference.

# Experiment Planning and Preprocessing

Bring the downloaded data into the [nnU-Net format](../../../nnUNet/documentation/dataset_format.md) and add the dataset.json file as given here:

```json
{
    "name": "Dataset311_FLARE24Task2_labeled",
    "description": "Abdominal Organ Segmentation",
    "labels": {
        "background": 0,
        "liver": 1,
        "right kidney": 2,
        "spleen": 3,
        "pancreas": 4,
        "aorta": 5,
        "ivc": 6,
        "rag": 7,
        "lag": 8,
        "gallbladder": 9,
        "esophagus": 10,
        "stomach": 11,
        "duodenum": 12,
        "left kidney": 13
    },
    "file_ending": ".nii.gz",
    "channel_names": {
        "0": "CT"
    },
    "overwrite_image_reader_writer": "NibabelIOWithReorient",
    "numTraining": 50
}
```

Afterwards you can run the default nnU-Net planning and preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 311 -c 3d_fullres
```

## Edit the plans files

The generated `nnUNetPlans.json` file needs to be edited to incorporate the custom low resolution setting.

```json
        "3d_halfres": {
            "inherits_from": "3d_fullres",
            "data_identifier": "nnUNetPlans_3d_halfres",
            "spacing": [
                5,
                1.6,
                1.6
            ]
        },
        "3d_halfiso": {
            "inherits_from": "3d_fullres",
            "data_identifier": "nnUNetPlans_3d_halfiso",
            "spacing": [
                2.5,
                2.5,
                2.5
            ]
        },
```

`3d_halfres` is a configuration with exactly half resolution, used as an ablation of our submission, `3d_halfiso` is the isotropic configuration we submitted as a final solution.

# Model training

Run one of the following commands to train the respective configurations. `3d_halfiso` yielded significantly better results in our experiments as well as on the final test set and is the recommended configuration.

```bash
nnUNetv2_train 311 3d_halfres all

nnUNetv2_train 311 3d_halfiso all
```

# Inference

Our inference is optimized for efficient single scan prediction. For best performance, we strongly recommend running inference using the default `nnUNetv2_predict` command!

Inference using the provided script requires OpenVINO, which can easily be installed via

```bash
pip install openvino
```

To run inference simply run the following commands. `model_folder` is the folder containing the training results, i.e. for example `nnUNetTrainer__nnUNetPlans__3d_halfiso`. `-save_model` needs to be set to precompile the model once using OpenVINO. If no precompiled model exists, the inference script will fail!

```bash
python inference_flare_task1.py -i input_folder -o output_folder -m model_folder [-save_model]
```
