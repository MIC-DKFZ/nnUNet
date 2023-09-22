# Look Ma, no code: fine tuning nnU-Net for the AutoPET II challenge by only adjusting its JSON plans

## Intro

See the [Challenge Website](https://autopet-ii.grand-challenge.org/) for details on the challenge.

We do not make any code adjustments to participate. All we do is optimize nnU-Net's hyperparameters 
(architecture, batch size, patch size) through modifying the nnUNetplans.json file.

## How to reproduce our trainings
Please read the [information on how to modify plans files](../explanation_plans_files.md) first!!!



## How to reproduce our results

1. Download the pretrained model weights from Zenodo LINK
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