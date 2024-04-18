# Residual Encoder Presets in nnU-Net

When using these presets, please cite our recent paper on the need for rigorous validation in 3D medical image segmentation:

> Isensee, F.<sup>* </sup>, Wald, T.<sup>* </sup>, Ulrich, C.<sup>* </sup>, Baumgartner, M.<sup>* </sup>, Roy, S., Maier-Hein, K.<sup>†</sup>, Jaeger, P.<sup>†</sup> (2024). nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. arXiv preprint arXiv:2404.09556.

*: shared first authors\
<sup>†</sup>: shared last authors

[PAPER LINK](https://arxiv.org/pdf/2404.09556.pdf)


Residual Encoder UNets have been supported by nnU-Net since our participation in KiTS2019, but have flown under the radar.
This is bound to change with our new nnUNetResEncUNet presets :raised_hands:! Especially on large datasets such as KiTS2023 and AMOS2022 
they offer improved segmentation performance!

|                        | BTCV  | ACDC  | LiTS  | BraTS | KiTS  | AMOS  |  VRAM |  RT | Arch. | nnU |
|------------------------|-------|-------|-------|-------|-------|-------|-------|-----|-------|-----|
|                        | n=30  | n=200 | n=131 | n=1251| n=489 | n=360 |       |     |       |     |
| nnU-Net (org.) [1]     | 83.08 | 91.54 | 80.09 | 91.24 | 86.04 | 88.64 |  7.70 |  9  |  CNN  | Yes |
| nnU-Net ResEnc M       | 83.31 | 91.99 | 80.75 | 91.26 | 86.79 | 88.77 |  9.10 |  12 |  CNN  | Yes |
| nnU-Net ResEnc L       | 83.35 | 91.69 | 81.60 | 91.13 | 88.17 | 89.41 | 22.70 |  35 |  CNN  | Yes |
| nnU-Net ResEnc XL      | 83.28 | 91.48 | 81.19 | 91.18 | 88.67 | 89.68 | 36.60 |  66 |  CNN  | Yes |
| MedNeXt L k3 [2]       | 84.70 | 92.65 | 82.14 | 91.35 | 88.25 | 89.62 | 17.30 |  68 |  CNN  | Yes |
| MedNeXt L k5 [2]       | 85.04 | 92.62 | 82.34 | 91.50 | 87.74 | 89.73 | 18.00 | 233 |  CNN  | Yes |
| STU-Net S [3]          | 82.92 | 91.04 | 78.50 | 90.55 | 84.93 | 88.08 |  5.20 |  10 |  CNN  | Yes |
| STU-Net B [3]          | 83.05 | 91.30 | 79.19 | 90.85 | 86.32 | 88.46 |  8.80 |  15 |  CNN  | Yes |
| STU-Net L [3]          | 83.36 | 91.31 | 80.31 | 91.26 | 85.84 | 89.34 | 26.50 |  51 |  CNN  | Yes |
| SwinUNETR [4]          | 78.89 | 91.29 | 76.50 | 90.68 | 81.27 | 83.81 | 13.10 |  15 |   TF  | Yes |
| SwinUNETRV2 [5]        | 80.85 | 92.01 | 77.85 | 90.74 | 84.14 | 86.24 | 13.40 |  15 |   TF  | Yes |
| nnFormer [6]           | 80.86 | 92.40 | 77.40 | 90.22 | 75.85 | 81.55 |  5.70 |  8  |   TF  | Yes |
| CoTr [7]               | 81.95 | 90.56 | 79.10 | 90.73 | 84.59 | 88.02 |  8.20 |  18 |   TF  | Yes |
| No-Mamba Base          | 83.69 | 91.89 | 80.57 | 91.26 | 85.98 | 89.04 |  12.0 |  24 |  CNN  | Yes |
| U-Mamba Bot [8]        | 83.51 | 91.79 | 80.40 | 91.26 | 86.22 | 89.13 | 12.40 |  24 |  Mam  | Yes |
| U-Mamba Enc [8]        | 82.41 | 91.22 | 80.27 | 90.91 | 86.34 | 88.38 | 24.90 |  47 |  Mam  | Yes |
| A3DS SegResNet [9,11]  | 80.69 | 90.69 | 79.28 | 90.79 | 81.11 | 87.27 | 20.00 |  22 |  CNN  |  No |
| A3DS DiNTS [10, 11]    | 78.18 | 82.97 | 69.05 | 87.75 | 65.28 | 82.35 | 29.20 |  16 |  CNN  |  No |
| A3DS SwinUNETR [4, 11] | 76.54 | 82.68 | 68.59 | 89.90 | 52.82 | 85.05 | 34.50 |  9  |   TF  |  No |

Results taken from our paper (see above), reported values are Dice scores computed over 5-fold cross-validation on each 
dataset. All models trained from scratch.

RT: training run time (measured on 1x Nvidia A100 PCIe 40GB)\
VRAM: GPU VRAM used during training, as reported by nvidia-smi\
Arch.: CNN = convolutional neural network; TF = transformer; Mam = Mamba\
nnU: whether the architectrue was integrated and tested with the nnU-Net framework (either by us or the original authors)

## How to use the new presets

We offer three new presets, each targeted for a different GPU VRAM and compute budget:
- **nnU-Net ResEnc M**: similar GPU budget to the standard UNet configuration. Best suited for GPUs with 9-11GB VRAM. Training time: ~12h on A100
- **nnU-Net ResEnc L**: requires a GPU with 24GB VRAM. Training time: ~35h on A100
- **nnU-Net ResEnc XL**: requires a GPU with 40GB VRAM. Training time: ~66h on A100

We recommend **nnU-Net ResEnc L** as the new default nnU-Net configuration to try!

The new presets are available as follows ((M/L/XL) = pick one!):
1. Specify the desired configuration when running experiment planning and preprocessing: 
`nnUNetv2_plan_and_preprocess -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)`. These planners use the same preprocessed
data folder as the standard 2d and 3d_fullres configurations since the preprocessed data is identical. Only the
3d_lowres differs and will be saved in a different folder to allow all configurations to coexist! If you are only 
planning to run 3d_fullres/2d and you already have this data preprocessed, you can just run 
`nnUNetv2_plan_experiment -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)` to avoid preprocessing again! 
2. Now, just specify the correct plans when running `nnUNetv2_train`, `nnUNetv2_predict` etc. The interface is 
consistent across all nnU-Net commands: `-p nnUNetResEncUNet(M/L/XL)Plans`  

Training results for the new presets will be stored in a dedicated folder and will not overwrite standard nnU-Net 
results! So don't be afraid to give it a go!

## Scaling ResEnc nnU-Net beyond the Presets
The presets differ from `ResEncUNetPlanner` in two ways:
- They set new default values for `gpu_memory_target_in_gb` to target the respective VRAM consumptions
- They remove the batch size cap of 0.05 (= previously one batch could not cover mode pixels than 5% of the entire dataset, not it can be arbitrarily large)

The preset are merely there ot make life easier, and to provide standardized configurations people can benchmark with.
You can easily adapt the GPU memory target to match your GPU, and to scale beyond 40GB of GPU memory. 

Here is an example for how to scale to 80GB VRAM on Dataset003_Liver:

`nnUNetv2_plan_experiment -d 3 -pl nnUNetPlannerResEncM -gpu_memory_target 80 -overwrite_plans_name nnUNetResEncUNetPlans_80G`

Just use `-p nnUNetResEncUNetPlans_80G` moving forward as outlined above! Running the example above will yield a 
warning ("You are running nnUNetPlannerM with a non-standard gpu_memory_target_in_gb"). This warning can be ignored here.
**Always change the plans identifier with `-overwrite_plans_name NEW_PLANS_NAME` when messing with the VRAM target in 
order to not overwrite preset plans!**

Why not use `ResEncUNetPlanner` -> because that one still has the 5% cap in place!

### Scaling to multiple GPUs
When scaling to multiple GPUs, do not just specify the combined amount of VRAM to `nnUNetv2_plan_experiment` as this 
may result in patch sizes that are too large to be processed by individual GPUs. It is best to let this command run for 
the VRAM budget of one GPU, and then manually edit the plans file to increase the batch size. You can use [configuration inheritance](explanation_plans_files.md).
In the configurations dictionary of the generated plans JSON file, add the following entry:

```json
        "3d_fullres_bsXX": {
            "inherits_from": "3d_fullres",
            "batch_size": XX
        },
```
Where XX is the new batch size. If 3d_fullres has a batch size of 2 for one GPU and you are planning to scale to 8 GPUs, make the new batch size 2x8=16!
You can then train the new configuration using nnU-Net's multi-GPU settings:

```bash
nnUNetv2_train DATASETID 3d_fullres_bsXX FOLD -p nnUNetResEncUNetPlans_80G -num_gpus 8
```

## Proposing a new segmentation method? Benchmark the right way!
When benchmarking new segmentation methods against nnU-Net, we encourage to benchmark against the residual encoder 
variants. For a fair comparison, pick the variant that most closely matches the GPU memory and compute 
requirements of your method!


## References
 [1] Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature methods 18.2 (2021): 203-211.\
 [2] Roy, Saikat, et al. "Mednext: transformer-driven scaling of convnets for medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.\
 [3] Huang, Ziyan, et al. "Stu-net: Scalable and transferable medical image segmentation models empowered by large-scale supervised pre-training." arXiv preprint arXiv:2304.06716 (2023).\
 [4] Hatamizadeh, Ali, et al. "Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images." International MICCAI Brainlesion Workshop. Cham: Springer International Publishing, 2021.\
 [5] He, Yufan, et al. "Swinunetr-v2: Stronger swin transformers with stagewise convolutions for 3d medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.\
 [6] Zhou, Hong-Yu, et al. "nnformer: Interleaved transformer for volumetric segmentation." arXiv preprint arXiv:2109.03201 (2021).\
 [7] Xie, Yutong, et al. "Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation." Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part III 24. Springer International Publishing, 2021.\
 [8] Ma, Jun, Feifei Li, and Bo Wang. "U-mamba: Enhancing long-range dependency for biomedical image segmentation." arXiv preprint arXiv:2401.04722 (2024).\
 [9] Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: 4th International Workshop, BrainLes 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Revised Selected Papers, Part II 4. Springer International Publishing, 2019.\
 [10] He, Yufan, et al. "Dints: Differentiable neural network topology search for 3d medical image segmentation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.\
 [11] Auto3DSeg, MONAI 1.3.0, [LINK](https://github.com/Project-MONAI/tutorials/tree/ed8854fa19faa49083f48abf25a2c30ab9ac1c6b/auto3dseg)

