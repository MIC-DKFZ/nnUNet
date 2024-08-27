# Introduction

This document describes our submission to the [Toothfairy2 Challenge](https://toothfairy2.grand-challenge.org/toothfairy2/). 
Our model is essentially a nnU-Net ResEnc L with the patch size upscaled to 160x320x320 pixels. We disable left/right 
mirroring and train for 1500 instead of the standard 1000 epochs. Training was either done on 2xA100 40GB or one GH200 96GB.

# Dataset Conversion

# Experiment Planning and Preprocessing

## Extract fingerprint:
`nnUNetv2_extract_fingerprint -d 119 -np 48`

## Run planning:
`nnUNetv2_plan_experiment -d 119 -pl nnUNetPlannerResEncL_torchres`

This planner not only uses the ResEncL configuration but also replaces the default resampling scheme with one that is 
faster (but less precise). Since all images in the challenge (train and test) should already have 0.3x0.3x0.3 spacing 
resampling is not required. This is just here as a safety measure. The speed is needed at inference time because grand 
challenge imposes a limit of 10 minutes per case.

## Edit the plans files
Add the following configuration to the generated plans file:

```json
        "3d_fullres_torchres_ps160x320x320_bs2": {
            "inherits_from": "3d_fullres",
            "patch_size": [
                160,
                320,
                320
            ],
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
                "arch_kwargs": {
                    "n_stages": 7,
                    "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320,
                        320
                    ],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ]
                    ],
                    "strides": [
                        [
                            1,
                            1,
                            1
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            1,
                            2,
                            2
                        ]
                    ],
                    "n_blocks_per_stage": [
                        1,
                        3,
                        4,
                        6,
                        6,
                        6,
                        6
                    ],
                    "n_conv_per_stage_decoder": [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1
                    ],
                    "conv_bias": true,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {
                        "eps": 1e-05,
                        "affine": true
                    },
                    "dropout_op": null,
                    "dropout_op_kwargs": null,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {
                        "inplace": true
                    }
                },
                "_kw_requires_import": [
                    "conv_op",
                    "norm_op",
                    "dropout_op",
                    "nonlin"
                ]
            }            
        }
```
Aside from changing the patch size this makes the architecture one stage deeper (one more pooling + res blocks), enabling
it to make effective use of the larger input

# Training
We train two models on all training cases:

```bash
nnUNetv2_train 119 3d_fullres_torchres_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_onlyMirror01_1500ep
nnUNet_results=${nnUNet_results}_2 nnUNetv2_train 119 3d_fullres_torchres_ps160x320x320_bs2 all -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_onlyMirror01_1500ep
```
Note how in the second line we overwrite the nnUNet_results variable in order to be able to train the same model twice without overwriting the results

# Inference
We ensemble the two models from above. On a technical level we copy the two fold_all folders into one training output 
directory and rename them to fold_0 and fold_1. This lets us use nnU-Net's cross-validation ensembling strategy which 
is more computationally efficient (needed for time limit on grand-challenge.org).

Run inference with the inference script