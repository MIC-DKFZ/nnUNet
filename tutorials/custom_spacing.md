Sometimes you want to set custom target spacings. This is done by creating a custom ExperimentPlanner.
Let's run this with the Task002_Heart example from the Medical Segmentation Decathlon. This dataset is not too large 
and working with it is therefore a breeze!

This example requires you to have downloaded the dataset and converted it to nnU-Net format with 
nnUNet_convert_decathlon_task

We need to run the nnUNet_plan_and_preprocess command with a custom 3d experiment planner to achieve this. I have 
created an appropriate trainer and placed it in `nnunet.experiment_planning.alternative_experiment_planning.target_spacing.experiment_planner_baseline_3DUNet_v21_customTargetSpacing_2x2x2.py`

This will set a hard coded target spacing of 2x2x2mm for the 3d_fullres configuration (3d_lowres is unchanged). 
Go have a look at this ExperimentPlanner now.

To run nnUNet_plan_and_preprocess with the new ExperimentPlanner, simply specify it:

`nnUNet_plan_and_preprocess -t 2 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_2x2x2`

Note how we are disabling 2D preprocessing with `-pl2d None`. The ExperimentPlanner I created is only for 3D. 
You will need to generate a separate one for 3D.

Once this is completed your task will have been preprocessed with the desired target spacing. You can use it by 
specifying the new custom plans file that is linked to it (see 
`ExperimentPlanner3D_v21_customTargetSpacing_2x2x2` source code) when running any nnUNet_* command, for example:

`nnUNet_train 3d_fullres nnUNetTrainerV2 2 FOLD -p nnUNetPlansv2.1_trgSp_2x2x2`

(make sure to omit the `_plans_3D.pkl` suffix!)

**TODO**: how to compare with the default run?

IMPORTANT: When creating custom ExperimentPlanner, make sure to always place them under a unique class name somewhere
in the nnunet.experiment_planning module. If you create subfolders, make sure they contain an __init__py file 
(can be empty). If you fail to do so nnU-Net will not be able to locate your ExperimentPlanner and crash!  