Authors: \
Maximilian Rokuss, Michael Baumgartner, Yannick Kirchhoff, Klaus H. Maier-Hein*, Fabian Isensee*

*: equal contribution

Author Affiliations:\
Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg \
Helmholtz Imaging

# Introduction

This document describes our submission to the [AortaSeg24 Challenge](hhttps://aortaseg24.grand-challenge.org/). 
Our model is essentially a nnU-Net ResEnc L with modified data augmentation. We disable left/right mirroring and use the heavy data augmentation [DA5 Trainer](../../nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerDA5.py). Training was performed on an A100 40GB GPU.

# Experiment Planning and Preprocessing
After converting the data into the [nnUNet format](../../../nnUNet/documentation/dataset_format.md) (either keep and just rename the .mha files or convert them to .nii.gz), you can run the preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d 610 -c 3d_fullres -pl nnUNetPlannerResEncL -np 16
```

# Training
We train our model using:

```bash
nnUNetv2_train 610 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_onlyMirror01_DA5
```
Models are trained from scratch. We train one model using all the images and a five fold cross validation ensemble for the submission.

We recommend to increase the number of processes used for data augmentation. Otherwise you can run into CPU bottlenecks.
Use `export nnUNet_n_proc_DA=32` or higher (if your system permits!).

# Inference
For inference you can use the default [nnUNet inference functionalities](../../../nnUNet/documentation/how_to_use_nnunet.md). Specifically, once the training is finished, run:

```bash
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER -f all
```

for the single model trained on all the data and 

```bash
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

for the five fold ensemble.