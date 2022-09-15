# AMOS 2022 challenge
This is how you can reproduce our results. We will provide the pretrained model weights soon so that you can also just 
use the trained models.

- Download the AMOS data
- Adapt paths in dataset conversion scripts and run them (the only notable difference is that we declare the input 
modality in task 1 to be CT and in task 2 we declare it as nonCT. This affects the intensity normalization): 
  - Task 1 (CT only) [Task216_Amos2022_task1](../../nnunet/dataset_conversion/Task216_Amos2022_task1.py) 
  - Task 2 (CT + MRI) [Task217_Amos2022_task2](../../nnunet/dataset_conversion/Task217_Amos2022_task2.py)
- Run planning and preprocessing. Due to the different configurations we have in our final solution, this will take a 
while and use some disk space
  - `nnUNet_plan_and_preprocess -t 216 217 -pl3d ExperimentPlanner3D_residual_v21_bfnnUNet_31 -pl2d None`
  - `nnUNet_plan_and_preprocess -t 216 -pl3d ExperimentPlanner3D_residual_v21_bfnnUNet -pl2d None`
  - `nnUNet_plan_and_preprocess -t 216 217 -pl3d ExperimentPlanner3D_residual_v21_bfnnUNet_31_spRegnnU -pl2d None`
  - at this point you might be confused and ask yourself what all that stuff is and why things are the way they 
  are and we really don't have a good answer for you here except that it was all over the placetowards the end 
  and well that's how we roll and we won with this so don't question us. haha
- Manually edit the plans files to increase the batch size. Read on how to do that [here](../tutorials/edit_plans_files.md):
  - Task216: 
    - edit plans `nnUNetPlans_bfnnUNet_fabresnet` to have batch size 5 for 3d_fullres. Save as `nnUNetPlans_bfnnUNet_fabresnet_bs5`
    - edit plans `nnUNetPlans_bfnnUNet_fabresnet_31` to have batch size 5 for 3d_fullres. Save as `nnUNetPlans_bfnnUNet_fabresnet_31_bs5`
    - edit plans `nnUNetPlans_bfnnUNet_fabresnet_31_spnnU` to have batch size 6 for 3d_fullres. Save as `nnUNetPlans_bfnnUNet_fabresnet_31_spnnU_bs6`
  - Task217:
    - edit plans `nnUNetPlans_bfnnUNet_fabresnet_31` to have batch size 5 for 3d_fullres. Save as `nnUNetPlans_bfnnUNet_fabresnet_31_bs5`
    - edit plans `nnUNetPlans_bfnnUNet_fabresnet_31_spnnU` to have batch size 6 for 3d_fullres. Save as `nnUNetPlans_bfnnUNet_fabresnet_31_spnnU_bs6`
  - Why exactly those batch sizes? They more or less exactly fit a 40GB A100 GPU. DDP in nnU-Net is pretty janky 
  so we avoid this in our reproduction code. Note that due to time constraints we actually used DDP for the 
  submission (same batch and everything as here, just more GPUs) 
- Now you can train, provided of course that you have 40GB or larger A100 GPUs. Or you know your way around DDP in 
nnU-Net. So realistically get some A100's.
  - For FOLD in range(5):
    - `nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_SimonsInit_DA5 216 FOLD -p nnUNetPlans_bfnnUNet_fabresnet_31_bs5`
    - `nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_SimonsInit 216 FOLD -p nnUNetPlans_bfnnUNet_fabresnet_31_spnnU_bs6`
    - `nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_SimonsInit 216 FOLD -p nnUNetPlans_bfnnUNet_fabresnet_bs5`
    - `nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_SimonsInit 217 FOLD -p nnUNetPlans_bfnnUNet_fabresnet_31_spnnU_bs6`
    - `nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_SimonsInit 217 FOLD -p nnUNetPlans_bfnnUNet_fabresnet_31_bs5`
- Great! You are several thousands kWh poorer. Now you can actually run inference (adapt the paths to your system):
  - Task1: [script](run_inference_task1.py)
  - Task2: [script](run_inference_task2.py)
  
  
### Additional information
- The inference scripts are exactly the same as the ones used in our docker containers. We just changed the paths. 

- Note that inference will take a while (we estimated 60h for the 200(240) train cases on an RTX 3090 - per Task!) and will 
produce a boatload of temporary files (like >1TB or so). The amount of temp file can be reduced by not processing all cases at once but instead in chunks while deleting no longer used softmax predictions. It's up to you to program that if you want this. 

- For the inference with the Docker containers we changed the `_internal_predict_3D_3Dconv_tiled` function of 
`SegmentationNetwork` to return GPU torch.Tensor for the softmax. This would be a breaking change for the rest of 
nnU-Net which why this is no longer the case here. To reproduce the full speed of our inference, change the following:
  - ```python
             if verbose: print("copying results to CPU")
 
             if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
 
            aggregated_results = aggregated_results.detach().cpu().numpy()
     ``` 
  to
  - ```python
             if verbose: print("copying results to CPU")
 
             if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach()
 
            aggregated_results = aggregated_results.detach()
    ```