# nnU-Net

In 3D biomedial image segmentation, dataset properties like imaging modality, image sizes, voxel spacings, class 
ratios etc vary drastically.
For example, images in the [Liver and Liver Tumor Segmentation Challenge dataset](https://competitions.codalab.org/competitions/17094) 
are computed tomography (CT) scans, about 512x512x512 voxels large, have isotropic voxel spacings and their 
intensity values are quantitative (Hounsfield Units).
The [Automated Cardiac Diagnosis Challenge dataset](https://acdc.creatis.insa-lyon.fr/) on the other hand shows cardiac 
structures in cine MRI with a typical image shape of 10x320x320 voxels, highly anisotropic voxel spacings and 
qualitative intensity values. In addition, the ACDC dataset suffers from slice misalignments and a heterogeneity of 
out-of-plane spacings which can cause severe interpolation artifacts of not handled properly. 

In current research practice, segmentation pipelines are designed manually and with one specific dataset in mind. 
Hereby, many pipeline settings depend directly or indirectly on the properties of the dataset 
and display a complex co-dependence: image size, for example, affects the patch size, which in 
turn affects the required receptive field of the network, a factor that itself influences several other 
hyperparameters in the pipeline. As a result, pipelines that were developed on on (type of) dataset are inherently 
incomaptible with other datasets in the domain.

**nnU-Net is the first segmentation method that is designed to deal with the dataset diversity found in the somain. It 
condenses and automates the keys decisions for designing a successful segmentation pipeline for any given dataset.**

nnU-Net makes the following contributions to the field:

1. **Standardized baseline:** nnU-Net is the first standardized deep learning benchmark in biomedical segmentation.
Without manual effort, researchers can compare their algorithms against nnU-Net on an arbitrary number of datasets 
to provide meaningful evidence for proposed improvements. 
2. **Out-of-the-box segmentation method:** nnU-Net is the first plug-and-play tool for state-of-the-art biomedical 
segmentation. Inexperienced users can use nnU-Net out of the box for their custom 3D segmentation problem without 
need for manual intervention. 
3. **Framework:** nnU-Net is a framework for fast and effective development of segmentation methods. Due to its modular 
structure, new architectures and methods can easily be integrated into nnU-Net. Researchers can then benefit from its
generic nature to roll out and evaluate their modifications on an arbitrary number of datasets in a 
standardized environment.  

For more information about nnU-Net, please read the following paper:

`Isensee, Fabian, et al. "nnU-Net: Breaking the Spell on Successful Medical Image Segmentation." arXiv preprint arXiv:1904.08128 (2019).` [NEEDS UPDATE]

Please also cite this paper if you are using nnU-Net for your research!


# Table of Contents
Todo, use https://ecotrust-canada.github.io/markdown-toc/

# Installation
nnU-Net is only tested on Linux (Ubuntu). It may work on other operating systems as well but we do not guarantee that it will.

nnU-Net requires a GPU! For inference, the GPU should have 4 GB of VRAM. For training nnU-Net models the GPU should have at 
least 11 GB (such as the RTX 2080ti). Due to the use of mixed precision, fastest training times are achieved with the 
Volta architecture (Titan V, V100 GPUs) (tensorcore acceleration for 3D convolutions does not yet work on Turing-based GPUs).

We recommend you run the following steps in a virtual environment. [Here is a quick how-to for Ubuntu.](documentation/virtualenv.md)

1) Install [PyTorch](https://pytorch.org/get-started/locally/)
2) Install [Nvidia Apex](https://github.com/NVIDIA/apex). Follow the instructions [here](https://github.com/NVIDIA/apex#quick-start). You can skip this step if all you want to do is run inference with 
our pretrained models.
3) Install nnU-Net depending on your use case:
    - For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running **inference with pretrained models**:
    
      ```pip install nnunet```
    
    - For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
      ```shell
      git clone https://github.com/MIC-DKFZ/nnUNet.git
      cd nnUNet
      pip install -e .
      ```
4) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. Please follow the 
instructions [here](documentation/setting_up_paths.md).

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net 
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNet_` for 
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this 
environment must be activated when executing the commands.

All nnU-Net commands have a `-h` option which gives information on how to use them.

# Usage

## How to run nnU-Net on a new datasets
Given some dataset, nnU-Net fully automatically configures an entire segmentation pipeline that matches its properties. 
nnU-Net covers the entire pipeline, from preprocessing to model configuration, model training, postprocessing 
all the way to ensembling. After running nnU-Net, the trained model(s) can be applied to the test cases for inference. 

### Dataset conversion
nnU-Net expects datasets in a structured format. This format closely (but not entirely) follows the data structure of 
the [Medical Segmentation Decthlon](http://medicaldecathlon.com/). Please read 
[this](documentation/dataset_conversion.md) for information on how to convert datasets to be compatible with nnU-Net.

### Experiment planning and preprocessing
As a first step, nnU-Net extracts a dataset fingerprint (a set of dataset-specific properties such as 
image sizes, voxel spacings, intensity information etc). This information is used to create three U-Net configurations: 
a 2D U-Net, a 3D U-Net that operated on full resolution images as well as a 3D U-Net cascade where the first U-Net 
creates a coarse segmentation map in downsampled images which is then refined by the second U-Net.

Provided that the requested raw dataset is located in the correct folder (`nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK`, 
also see [here](documentation/dataset_conversion.md)), you can run this step with the following command:

```bash
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
```

`XXX` is the integer identifier associated with your Task name `TaskXXX_MYTASK`. You can pass several task IDs at once.

`--verify_dataset_integrity` should be run at least for the first time the command is run on a given dataset. This will execute some
 checks on the dataset to ensure that it is compatible with nnU-Net. If this check has passed once, it can be 
omitted in future runs.

Note that `nnUNet_plan_and_preprocess` accepts several additional input arguments. Running `-h` will list all of them 
along with a description. If you run out of RAM during preprocessing, you may want to adapt the number of processes 
used with the `-tl` and `-tf` options.

After `nnUNet_plan_and_preprocess` is completed, the U-Net configurations have been created and a preprocessed copy 
of the data will be located at nnUNet_preprocessed/TaskXXX_MYTASK.

### Model training
nnU-Net trains all U-Net configurations in a 5-fold cross-validation. This enables nnU-Net to determine the 
postprocessing and ensembling (see next step) on the training dataset. Per default, all U-Net configurations need to 
be run on a given dataset. There are, however situations in which only some configurations (and maybe even without 
running the cross-validation) are desired. See **TODO** for more information.

Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the U-Net 
cascade is omitted because the patch size of the full resolution U-Net already covers a large part of the input images.

Training models is done with the `nnUNet_train` command. The general structure of the command is:
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD (additional options)
```

CONFIGURATION is a string that identifies the requested U-Net configuration. TRAINER_CLASS_NAME is the name of the 
model trainer. If you implement custom trainers (nnU-Net as a framework) you can specify your custom trainer here.
TASK_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of the 5-fold-crossvalidaton is trained.

#### 2D U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 2d nnUNetTrainerV2 TaskXXX_MYTASK FOLD
```

#### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 TaskXXX_MYTASK FOLD
```

#### 3D U-Net cascade
##### 3D low resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_lowres nnUNetTrainerV2 TaskXXX_MYTASK FOLD
```

##### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes TaskXXX_MYTASK FOLD
```

Note that the 3D full resolution U-Net of the cascade requires the five folds of the low resolution U-Net to be completed beforehand!

### Identifying the best U-Net configuration(s)
Once all models are trained, use the following command to automatically determine what U-Net configuration(s) to use for test set prediction:

```bash
nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t XXX --allow_missing_pp --strict
```

On datasets for which the cascade was not configured, use `-m 2d 3d_fullres` instead. If you wish to only explore some 
subset of the configurations, you can specify that with the `-m` command. We recommend setting the 
`--allow_missing_pp` (determines postprocessing) and `--strict` (crash if one of the requested configurations is 
missing) flags. Additional options are available (use `-h` for help).

### Run inference
Remember that the data located in the input folder must adhere to the format specified 
[here](documentation/data_format_inference.md). 

`nnUNet_find_best_configuration` will print a string to the terminal with the inference commands you need to use. 
The easiest way to run inference is to simply use these commands. 

If you wish to manually specify the configuration(s) used for inference, use the following commands:

For each of the desired configurations, run:
```
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz
```

Only specify `--save_npz` if you intend to use ensembling. `--save_npz` will make the command save the softmax 
probabilities alongside of the predicted segmentation masks requiring a lot of disk space.

Please select a separate `OUTPUT_FOLDER` for each configuration!

If you wish to run ensembling, you can ensemble the predictions from several configurations with the following command:
```bash
nnUNet_ensemble -f FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -pp POSTPROCESSING_FILE
```

You can specify an arbitrary number of folders, but remember that each folder needs to contain npz files that were 
generated by `nnUNet_predict`. For ensembling you can also specify a file that tells the command how to postprocess. 
These files are created when running `nnUNet_find_best_configuration` and are located in the respective trained model 
directory (RESULTS_FOLDER/nnUNet/CONFIGURATION/TaskXXX_MYTASK/TRAINER_CLASS_NAME__PLANS_FILE_IDENTIFIER/postprocessing.json or 
RESULTS_FOLDER/nnUNet/ensembles/TaskXXX_MYTASK/ensemble_X__Y__Z--X__Y__Z/postprocessing.json). You can also choose to 
not provide a file (simply omit -pp) and nnU-Net will not run postprocessing.

## How to run inference with pretrained models
TODO, depends on how I upload the models

# Extending nnU-Net
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




These 
#
for changes in training, crearte your own trainer class

for changes in preprocessing, create your own preprocessor and link it in the experiment planner

for changes to architecture and experiment planning, create your own experiment planner

# FAQ
#### Manual Splitting of Data
The cross-validation in nnU-Net splits on a per-case basis. This may sometimes not be desired, for example because 
several training cases may be the same patient (different time steps or annotators). If this is the case, then you need to
manually create a split file. To do this, first let nnU-Net create the default split file. Run one of the network 
trainings (any of them works fine for this) and abort after the first epoch. nnU-Net will have created a split file automatically:
`preprocessing_output_dir/TaskXX_MY_DATASET/splits_final.pkl`. This file contains a list (length 5, one entry per fold). 
Each entry in the list is a dictionary with keys 'train' and 'val' pointing to the patientIDs assigned to the sets. 
To use your own splits in nnU-Net, you need to edit these entries to what you want them to be and save it back to the 
splits_final.pkl file. Use load_pickle and save_pickle from batchgenerators.utilities.file_and_folder_operations for convenience.

#### Do I need to always run all U-Net configurations?
The model training pipeline above is for challenge participations. Depending on your task you may not want to train all 
U-Net models and you may also not want to run a cross-validation all the time.
Here are some recommendations about what U-Net model to train:
- It is safe to say that on average, the 3D U-Net model (3d_fullres) was most robust. If you just want to use nnU-Net because you 
need segmentations, I recommend you start with this.
- If you are not happy with the results from the 3D U-Net then you can try the following:
  - if your cases are very large so that the patch size of the 3d U-Net only covers a very small fraction of an image then 
  it is possible that the 3d U-Net cannot capture sufficient contextual information in order to be effective. If this 
  is the case, you should consider running the 3d U-Net cascade (3d_lowres followed by 3d_cascade_fullres)
  - If your data is very anisotropic then a 2D U-Net may actually be a better choice (Promise12, ACDC, Task05_Prostate 
  from the decathlon are good examples)

You do not have to run five-fold cross-validation all the time. If you want to test single model performance, use
 *all* for `FOLD` instead of a number.
 
CAREFUL: DO NOT use fold=all when you intend to run the cascade! You must run the cross-validation in 3d_lowres so that you get proper (=not overfitted) low resolution predictions.
 
#### Sharing Models
You can share trained models by simply sending the corresponding output folder from `RESULTS_FOLDER/nnUNet` to 
whoever you want share them with. The recipient can then use nnU-Net for inference with this model.

#### Can I use multi GPU training?
TODO

#### Can I run nnU-Net on smaller GPUs?
TODO

#### I get the error `seg from prev stage missing` when running the cascade
You need to run all five folds of `3d_lowres`. Segmentations of the previous stage can only be generated from the 
validation set, otherwise we would overfit.

#### Why am I getting `RuntimeError: CUDA error: device-side assert triggered`?
This error often goes along with something like `void THCudaTensor_scatterFillKernel(TensorInfo<Real, IndexType>, 
TensorInfo<long, IndexType>, Real, int, IndexType) [with IndexType = unsigned int, Real = float, Dims = -1]: 
block: [4770,0,0], thread: [374,0,0] Assertion indexValue >= 0 && indexValue < tensor.sizes[dim] failed.`.

This means that your dataset contains unexpected values in the segmentations. nnU-Net expects all labels to be 
consecutive integers. So if your dataset has 4 classes (background and three foregound labels), then the labels 
must be 0, 1, 2, 3 (where 0 must be background!). There cannot be any other values in the ground truth segmentations.

If you run `nnUNet_plan_and_preprocess` with the --verify_dataset_integrity option, this should never happen because 
it will check for wrong values in the label images.

#### Why is no 3d_lowres model created?
3d_lowres is created only if the patch size in 3d_fullres less than 1/4 of the voxels of the median shape of the data 
in 3d_fullres (for example Liver is about 512x512x512 and the patch size is 128x128x128, so that's 1/64 and thus 
3d_lowres is created). You can enforce the creation of 3d_lowres models for smaller datasets by changing the value of
`HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0` (located in experiment_planning.configuration).
    
