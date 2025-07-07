# Look Ma, no code: fine tuning nnU-Net for the AutoPET II challenge by only adjusting its JSON plans

Please cite our paper :-*

```text
COMING SOON
```

## Intro

See the [Challenge Website](https://autopet-ii.grand-challenge.org/) for details on the challenge.

Our solution to this challenge rewuires no code changes at all. All we do is optimize nnU-Net's hyperparameters 
(architecture, batch size, patch size) through modifying the nnUNetplans.json file.

## Prerequisites
Use the latest pytorch version!

We recommend you use the latest nnU-Net version as well! We ran our trainings with commit 913705f which you can try in case something doesn't work as expected:
`pip install git+https://github.com/MIC-DKFZ/nnUNet.git@913705f`

## How to reproduce our trainings

### Download and convert the data
1. Download and extract the AutoPET II dataset
2. Convert it to nnU-Net format by running `python nnunetv2/dataset_conversion/Dataset221_AutoPETII_2023.py FOLDER` where folder is the extracted AutoPET II dataset.

### Experiment planning and preprocessing
We deviate a little from the standard nnU-Net procedure because all our experiments are based on just the 3d_fullres configuration

Run the following commands:
   - `nnUNetv2_extract_fingerprint -d 221` extracts the dataset fingerprint 
   - `nnUNetv2_plan_experiment -d 221` does the planning for the plain unet
   - `nnUNetv2_plan_experiment -d 221 -pl ResEncUNetPlanner` does the planning for the residual encoder unet
   - `nnUNetv2_preprocess -d 221 -c 3d_fullres` runs all the preprocessing we need

### Modification of plans files
Please read the [information on how to modify plans files](../explanation_plans_files.md) first!!!


It is easier to have everything in one plans file, so the first thing we do is transfer the ResEnc U-Net to the 
default plans file. We use the configuration inheritance feature of nnU-Net to make it use the same data as the 
3d_fullres configuration.
Add the following to the 'configurations' dict in 'nnUNetPlans.json':

```json
        "3d_fullres_resenc": {
            "inherits_from": "3d_fullres",
            "network_arch_class_name": "ResidualEncoderUNet",
            "n_conv_per_stage_encoder": [
                1,
                3,
                4,
                6,
                6,
                6
            ],
            "n_conv_per_stage_decoder": [
                1,
                1,
                1,
                1,
                1
            ]
        },
```

(these values are basically just copied from the 'nnUNetResEncUNetPlans.json' file! With everything redundant being omitted thanks to inheritance from 3d_fullres)

Now we crank up the patch and batch sizes. Add the following configurations:
```json
        "3d_fullres_resenc_bs80": {
            "inherits_from": "3d_fullres_resenc",
            "batch_size": 80
            },
        "3d_fullres_resenc_192x192x192_b24": {
            "inherits_from": "3d_fullres_resenc",
            "patch_size": [
                192,
                192,
                192
            ],
            "batch_size": 24
        }
```

Save the file (and check for potential Syntax Errors!)

### Run trainings
Training each model requires 8 Nvidia A100 40GB GPUs. Expect training to run for 5-7 days. You'll need a really good 
CPU to handle the data augmentation! 128C/256T are a must! If you have less threads available, scale down nnUNet_n_proc_DA accordingly.

```bash
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 0 -num_gpus 8
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 1 -num_gpus 8
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 2 -num_gpus 8
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 3 -num_gpus 8
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 4 -num_gpus 8

nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 0 -num_gpus 8
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 1 -num_gpus 8
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 2 -num_gpus 8
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 3 -num_gpus 8
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 4 -num_gpus 8
```

Done!

(We also provide pretrained weights in case you don't want to invest the GPU resources, see below)

## How to make predictions with pretrained weights
Our final model is an ensemble of two configurations:
- ResEnc U-Net with batch size 80
- ResEnc U-Net with patch size 192x192x192 and batch size 24

To run inference with these models, do the following:

1. Download the pretrained model weights from [Zenodo](https://zenodo.org/record/8362371)
2. Install both .zip files using `nnUNetv2_install_pretrained_model_from_zip`
3. Make sure 
4. Now you can run inference on new cases with `nnUNetv2_predict`:
   - `nnUNetv2_predict -i INPUT -o OUTPUT1 -d 221 -c 3d_fullres_resenc_bs80 -f 0 1 2 3 4 -step_size 0.6 --save_probabilities`   
   - `nnUNetv2_predict -i INPUT -o OUTPUT2 -d 221 -c 3d_fullres_resenc_192x192x192_b24 -f 0 1 2 3 4 --save_probabilities`
   - `nnUNetv2_ensemble -i OUTPUT1 OUTPUT2 -o OUTPUT_ENSEMBLE`

Note that our inference Docker omitted TTA via mirroring along the axial direction during prediction (only sagittal + 
coronal mirroring). This was
done to keep the inference time below 10 minutes per image on a T4 GPU (we actually never tested whether we could 
have left this enabled). Just leave it on! You can also leave the step_size at default for the 3d_fullres_resenc_bs80.