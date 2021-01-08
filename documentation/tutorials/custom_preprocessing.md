When you would like to interfere with the way resampling during preprocessing is handled or you would like to implement 
a custom normalization scheme, you need to create a new custom preprocessor class and an ExperimentPlanner to go along 
with it. While this may appear cumbersome, the great thing about this approach is that the same code will be used for 
inference as well thus guaranteeing that images are preprocessed properly (i.e. the way the model expects).

In this tutorial we will implement a custom normalization scheme for the Task120 Massachusetts Road Segmentation. Make 
sure to download the dataset and run the code in nnunet.dataset_conversion.Task120_Massachusetts_RoadSegm.py prior to this tutorial.

The images in the dataset are RGB with a value range of [0, 255]. nnU-nets defaultnormalization scheme will normalize 
each color channel independently to have mean 0 and standard deviation 1. This works reasonably well, but may result 
in a shift of the color channels relative to each other and thus disturb the models performance. To address that, the new
normalization will rescale the value range from [0, 255] to [0, 1] by simply dividing the intensities of each image by 
255. Thus, there will be no longer a shift between the color channels.

The new preprocessor class is located in nnunet.preprocessing.custom_preprocessors.preprocessor_scale_RGB_to_0_1.py. 
To acutally use it, we need to tell the ExperimentPlanner its name. For this purpose, it is best to create a new 
ExperimentPlanner class. I created one and placed it in 
nnunet.experiment_planning.alternative_experiment_planning.normalization.experiment_planner_2DUNet_v21_RGB_scaleto_0_1.py

Now go have a look at these two classes. Details are in the comments there.

To run the new preprocessor, you need to specify its accompanying ExperimentPlanner when running 
`nnUNet_plan_and_preprocess`:

```bash
nnUNet_plan_and_preprocess -t 120 -pl3d None -pl2d ExperimentPlanner2D_v21_RGB_scaleTo_0_1
```

After that you can run the training:

```bash
nnUNet_train 2d nnUNetTrainerV2 120 FOLD -p nnUNet_RGB_scaleTo_0_1
```

Note that `nnUNet_RGB_scaleTo_0_1` is the plans identifier defined in our custom ExperimentPlanner. Specify it for all 
nnUNet_* commands whenever you want to use the models resulting from this training.

Remember to always place custom ExperimentPlanner in nnunet.experiment_planning (any file or submodule) and 
preprocessors in nnunet.preprocessing (any file or submodule). Make sure to use unique names!

The example classes from this tutorial only work with 2D. You need to generate a separate set of planner and preprocessor
for 3D data (cumbersome, I know. Needs to be improved in the future).