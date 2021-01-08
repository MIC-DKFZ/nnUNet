
# Extending/Changing nnU-Net

To use nnU-Net as a framework and make changes to its components, please make sure to install it with the `git clone` 
and `pip install -e .` commands so that a local copy of the code is created.
Changing components of nnU-Net needs to be done in different places, depending on whether these components belong to 
the inferred, blueprint or empirical parameters. We cover some of the most common use cases below. They should give 
you a good indication of where to start.

Generally it is recommended to look into the code where the thing you would like to change is currently implemented 
and then derive a strategy on how to change it. If you have any questions, feel free to open an issue on GitHub and 
we will help you as much as we can.

## Changes to blueprint parameters
This section gives guidance on how to implement changes to loss function, training schedule, learning rates, optimizer, 
some architecture parameters, data augmentation etc. All these parameters are part of the **nnU-Net trainer class**, 
which we have already seen in the sections above. The default trainer class for 2D, 3D low resolution and 3D full 
resolution U-Net is nnUNetTrainerV2, the default for the 3D full resolution U-Net from the cascade is 
nnUNetTrainerV2CascadeFullRes. Trainer classes in nnU-Net inherit form each other, nnUNetTrainerV2CascadeFullRes for 
example has nnUNetTrainerV2 as parent class and only overrides cascade-specific code.

Due to the inheritance of trainer classes, changes can be integrated into nnU-Net quite easily and with minimal effort. 
Simply create a new trainer class (with some custom name), change the functionality you need to change and then specify 
this class (via its name) during training - done.

This process requires the new class to be located in a subfolder of nnunet.training.network_training! Do not save it 
somewhere else or nnU-Net will not be able to find it! Also don't use the same name twice! nnU-Net always picks the 
first trainer that matches the requested name.

Don't worry about overwriting results of another trainer class. nnU-Net always generates output folders that are named 
after the trainer class used to generate the results. 

Due to the variety of possible changes to the blueprint parameters of nnU-Net, we here only present a summary of where 
to look for what kind of modification. During method development we have already created a large number of nnU-Net 
blueprint variations which should give a good indication of where to start:

| Type of modification    | Examples                                                                                                                                                                                                                                                                                                                                              |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| loss function           | nnunet.training.network_training.loss_function.*                                                                                                                                                                                                                                                                                                      |
| data augmentation       | nnunet.training.network_training.data_augmentation.*                                                                                                                                                                                                                                                                                                  |
| Optimizer, lr, momentum | nnunet.training.network_training.optimizer_and_lr.*                                                                                                                                                                                                                                                                                                   |
| (Batch)Normalization    | nnunet.training.network_training.architectural_variants.nnUNetTrainerV2_BN.py<br>nnunet.training.network_training.architectural_variants.nnUNetTrainerV2_FRN.py<br>nnunet.training.network_training.architectural_variants.nnUNetTrainerV2_GN.py<br>nnunet.training.network_training.architectural_variants.nnUNetTrainerV2_NoNormalization_lr1en3.py |
| Nonlinearity            | nnunet.training.network_training.architectural_variants.nnUNetTrainerV2_ReLU.py<br>nnunet.training.network_training.architectural_variants.nnUNetTrainerV2_Mish.py                                                                                                                                                                                    |
| Architecture            | nnunet.training.network_training.architectural_variants.nnUNetTrainerV2_3ConvPerStage.py<br>nnunet.training.network_training.architectural_variants.nnUNetTrainerV2_ResencUNet                                                                                                                                                                        |
| ...                     | (see nnunet.training.network_training and subfolders)                                                                                                                                                                                                                                                                                                 |

## Changes to Inferred Parameters
The inferred parameters are determined based on the dataset fingerprint, a low dimensional representation of the properties 
of the training cases. It captures, for example, the image shapes, voxel spacings and intensity information from 
the training cases. The datset fingerprint is created by the DatasetAnalyzer (which is located in nnunet.preprocessing) 
while running `nnUNet_plan_and_preprocess`. 

`nnUNet_plan_and_preprocess` uses so called ExperimentPlanners for running the adaptation process. Default ExperimentPlanner 
classes are ExperimentPlanner2D_v21 for the 2D U-Net and ExperimentPlanner3D_v21 for the 3D full resolution U-Net and the 
U-Net cascade. Just like nnUNetTrainers, the ExperimentPlanners inherit from each other, resulting in minimal programming 
effort to incorporate changes. Just like with the trainers, simply give your custom ExperimentPlanners a unique name and 
save them in some subfolder of nnunet.experiment_planning. You can then specify your class names when running 
`nnUNet_plan_and_preprocess` and nnU-Net will find them automatically. When inheriting form ExperimentPlanners, you **MUST** 
overwrite the class variables `self.data_identifier` and `self.plans_fname` (just like for example 
[here](../nnunet/experiment_planning/alternative_experiment_planning/normalization/experiment_planner_3DUNet_CT2.py)). 
If you omit this step the planner will overwrite the plans file and the preprocessed data of the planner it inherits from.

To train with your custom configuration, simply specify the correct plans identifier with `-p` when you call the 
`nnUNet_train` command. The plans file also contains the data_identifier specified in your ExperimentPlanner, so the 
trainer class will automatically know what data should be used.

Possible adaptations to the inferred parameters could include a different way of prioritizing batch size vs patch size 
(currently, nnU-Net prioritizies patch size), a different handling of the spacing information for architecture template 
instantiation, changing the definition of target spacing, or using different strategies for finding the 3d low 
resolution U-Net configuration.

The folders located in nnunet.experiment_planning contain several example ExperimentPlanner that modify various aspects 
of the inferred parameters. You can use them as inspiration for your own.

If you wish to run a different preprocessing, you most likely will have to implement your own Preprocessor class. 
The preprocessor class that is used by some ExperimentPlanner is specified in its preprocessor_name class variable. The 
default is `self.preprocessor_name = "GenericPreprocessor"` for 3D and `PreprocessorFor2D` for 2D (the 2D preprocessor 
ignores the target spacing for the first axis to ensure that images are only resampled in the axes that will make up the training samples). 
GenericPreprocessor (and all custom Preprocessors you implement) must be located in nnunet.preprocessing. The 
preprocessor_name is saved in the plans file (by ExperimentPlanner), so that the 
nnUNetTrainer knows which preprocessor must be used during inference to match the preprocessing of the training data. 

Modifications to the preprocessing pipeline could be the addition of bias field correction to MRI images, a different CT
preprocessing scheme or a different way of resampling segmentations and image data for anisotropic cases. 
An example is provided [here](../nnunet/preprocessing/preprocessing.py).

When implementing a custom preprocessor, you should also create a custom ExperimentPlanner that uses it (via self.preprocessor_name). 
This experiment planner must also use a matching data_identifier and plans_fname to ensure no other data is overwritten.

## Use a different network architecture
Changing the network architecture in nnU-Net is easy, but not self-explanatory. Any new segmentation network you implement 
needs to understand what nnU-Net requests from it (wrt how many downsampling operations are done, whether deep supervision 
is used, what the convolutional kernel sizes are supposed to be). It needs to be able to dynamiccaly change its topology, 
just like our implementation of the [Generic_UNet](../nnunet/network_architecture/generic_UNet.py). Furthermore, it must be
able to generate a value that can be used to estimate memory consumption. What we have implemented for Generic_UNet effectively
counts the number of voxels found in all feature maps that are present in a given configuration. Although this estimation 
disregards the number of parameters we have found it to work quite well. Unless you implement an architecture with 
unreasonably high number of parameters, the large majority of the VRAM used during training will be occupied by feature 
maps, so parameters can be (mostly) disregarded. For implementing your own network, it is key to understand that the 
number we are computing here cannot be interpreted directly as memory consumption (other factors than the feature maps 
of the convolutions also play a role, such as instance normalization. This is furthermore very hard to predict because 
there are also several different algorithms for running the convolutions, each with its own memory requirement. We train 
models with cudnn.benchmark=True, so it is impossible to predict which algorithm is used). 
So instead, to approch this problem in the most straightforward way, we manually identify the largest configuration we 
can fit in the GPU of choice (manually define the dowmsampling, patch size etc) and use this value (-10% or so to be save) 
as **reference** in the ExperimentPlanner that uses this architecture. 

To illustrate this process, we have implemented a U-Net with a residual encoder 
(see FabiansUNet in [generic_modular_residual_UNet.py](../nnunet/network_architecture/generic_modular_residual_UNet.py)). 
This UNet has a class variable called use_this_for_3D_configuration. This value was found with the code located in 
find_3d_configuration (same python file). The corresponding ExperimentPlanner 
[ExperimentPlanner3DFabiansResUNet_v21](../nnunet/experiment_planning/alternative_experiment_planning/experiment_planner_residual_3DUNet_v21.py)
compares this value to values generated for the currently configured network topology (which are also computed by 
FabiansUNet.compute_approx_vram_consumption) to ensure that the GPU memory target is met.

## Tutorials
We have created tutorials on how to [manually edit plans files](tutorials/edit_plans_files.md), 
[change the target spacing](tutorials/custom_spacing.md) and 
[changing the normalization scheme for preprocessing](tutorials/custom_preprocessing.md).