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
- [Installation](#installation)
- [Usage](#usage)
  * [How to run nnU-Net on a new datasets](#how-to-run-nnu-net-on-a-new-datasets)
    + [Dataset conversion](#dataset-conversion)
    + [Experiment planning and preprocessing](#experiment-planning-and-preprocessing)
    + [Model training](#model-training)
      - [2D U-Net](#2d-u-net)
      - [3D full resolution U-Net](#3d-full-resolution-u-net)
      - [3D U-Net cascade](#3d-u-net-cascade)
        * [3D low resolution U-Net](#3d-low-resolution-u-net)
        * [3D full resolution U-Net](#3d-full-resolution-u-net-1)
      - [Multi GPU training](#multi-gpu-training)
    + [Identifying the best U-Net configuration(s)](#identifying-the-best-u-net-configuration-s-)
    + [Run inference](#run-inference)
  * [How to run inference with pretrained models](#how-to-run-inference-with-pretrained-models)
- [Extending/Changing nnU-Net](#extending-changing-nnu-net)
- [FAQ](#faq)

ecotrust-canada.github.io/markdown-toc/

# Installation
nnU-Net is only tested on Linux (Ubuntu). It may work on other operating systems as well but we do not guarantee that it will.

nnU-Net requires a GPU! For inference, the GPU should have 4 GB of VRAM. For training nnU-Net models the GPU should have at 
least 11 GB (such as the RTX 2080ti). Due to the use of mixed precision, fastest training times are achieved with the 
Volta architecture (Titan V, V100 GPUs) (tensorcore acceleration for 3D convolutions does not yet work on Turing-based GPUs).

We recommend you run the following steps in a virtual environment. [Here is a quick how-to for Ubuntu.](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/)

1) Install [PyTorch](https://pytorch.org/get-started/locally/)
2) Install [Nvidia Apex](https://github.com/NVIDIA/apex). Follow the instructions [here](https://github.com/NVIDIA/apex#quick-start).
You can skip this step if all you want to do is run inference with our pretrained models. Apex is required for 
mixed precision training. (Please **do not use** `pip install apex` - this will not install the correct package). 
When installing apex, you have two choices (both are described on the apex website linked above!):
    1) Python-only installation:
    This will not compile custom kernels and is a little bit slower than the other option (<10%). But it is much easier to do, 
    which is why we recommend this option for less experienced users
    2) Regular installation:
    This gives more performance, but requires a CUDA toolkit installation. When installing pytorch, you must make sure to 
    select the Cuda version that matches the toolkit version you have installed. You can check which version you have by 
    running `nvcc --version`. You furthermore need to have the python3 dev libraries installed on your system. Follow the
    instructions [here](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory) for how to do this.
    Only after these prerequisites are done you can install apex.
    Note that pytorch will compile the kernels only for the type of GPU that is in your system. If you intend to swap 
    out your GPU (or are installing this in a cluster environment), run `export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5"` 
    prior to installing apex. This will tell pytorch to compile for all currently available GPU types.
    
3) Install nnU-Net depending on your use case:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running **inference with pretrained models**:
      
        ```pip install nnunet```
    
    2) For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
4) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to 
set a few of environment variables. Please follow the instructions [here](documentation/setting_up_paths.md).
5) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate 
plots of the network topologies it generates (see [Model training](#model-training)). To install hiddenlayer, 
run the following commands:
    ```bash
    git clone https://github.com/nanohanno/hiddenlayer.git
    cd hiddenlayer
    git checkout bugfix/get_trace_graph
    pip install -e .
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net 
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNet_` for 
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this 
environment must be activated when executing the commands.

All nnU-Net commands have a `-h` option which gives information on how to use them.

# Usage

## How to run nnU-Net on a new dataset
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

Running `nnUNet_plan_and_preprocess` will populate your folder with preprocessed data. You will find the output in 
nnUNet_preprocessed/TaskXXX_MYTASK. `nnUNet_plan_and_preprocess` creates subfolders with preprocessed data for the 2D 
U-Net as well as all applicable 3D U-Nets. It will also create 'plans' files (with the ending.pkl) for the 2D and 
3D configurations. These files contain the generated segmentation pipeline configuration and will be read by the 
nnUNetTrainer (see below). Note that the preprocessed data folder only contains the training cases. 
The test images are not preprocessed (they are not looked at at all!). Their preprocessing happens on the fly during 
inference.

`--verify_dataset_integrity` should be run at least for the first time the command is run on a given dataset. This will execute some
 checks on the dataset to ensure that it is compatible with nnU-Net. If this check has passed once, it can be 
omitted in future runs. If you adhere to the dataset conversion guide (see above) then this should pass without issues :-)

Note that `nnUNet_plan_and_preprocess` accepts several additional input arguments. Running `-h` will list all of them 
along with a description. If you run out of RAM during preprocessing, you may want to adapt the number of processes 
used with the `-tl` and `-tf` options.

After `nnUNet_plan_and_preprocess` is completed, the U-Net configurations have been created and a preprocessed copy 
of the data will be located at nnUNet_preprocessed/TaskXXX_MYTASK.

### Model training
nnU-Net trains all U-Net configurations in a 5-fold cross-validation. This enables nnU-Net to determine the 
postprocessing and ensembling (see next step) on the training dataset. Per default, all U-Net configurations need to 
be run on a given dataset. There are, however situations in which only some configurations (and maybe even without 
running the cross-validation) are desired. See [FAQ](#faq) for more information.

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

Note that the 3D full resolution U-Net of the cascade requires the five folds of the low resolution U-Net to be 
completed beforehand!

The trained models will we written to the RESULTS_FOLDER/nnUNet folder. Each training obtains an automatically generated 
output folder name:

nnUNet_preprocessed/CONFIGURATION/TaskXXX_MYTASKNAME/TRAINER_CLASS_NAME__PLANS_FILE_NAME/FOLD

For Task002_Heart (from the MSD), for example, this looks like this:

    RESULTS_FOLDER/nnUNet/
    ├── 2d
    │   └── Task02_Heart
    │       └── nnUNetTrainerV2__nnUNetPlansv2.1
    │           ├── fold_0
    │           ├── fold_1
    │           ├── fold_2
    │           ├── fold_3
    │           └── fold_4
    ├── 3d_cascade_fullres
    ├── 3d_fullres
    │   └── Task02_Heart
    │       └── nnUNetTrainerV2__nnUNetPlansv2.1
    │           ├── fold_0
    │           │   ├── debug.json
    │           │   ├── model_best.model
    │           │   ├── model_best.model.pkl
    │           │   ├── model_final_checkpoint.model
    │           │   ├── model_final_checkpoint.model.pkl
    │           │   ├── network_architecture.pdf
    │           │   ├── progress.png
    │           │   └── validation_raw
    │           │       ├── la_007.nii.gz
    │           │       ├── la_007.pkl
    │           │       ├── la_016.nii.gz
    │           │       ├── la_016.pkl
    │           │       ├── la_021.nii.gz
    │           │       ├── la_021.pkl
    │           │       ├── la_024.nii.gz
    │           │       ├── la_024.pkl
    │           │       ├── summary.json
    │           │       └── validation_args.json
    │           ├── fold_1
    │           ├── fold_2
    │           ├── fold_3
    │           └── fold_4
    └── 3d_lowres


Note that 3d_lowres and 3d_cascade_fullres are not populated because this dataset did not trigger the cascade. In each 
model training output folder (each of the fold_x folder, 10 in total here), the following files will be created (only 
shown for one folder above for brevity):
- debug.json: Contains a summary of blueprint and inferred parameters used for training this model. Not easy to read, 
but very useful for debugging ;-)
- model_best.model / model_best.model.pkl: checkpoint files of the best model identified during training. Not used right now.
- model_final_checkpoint.model / model_final_checkpoint.model.pkl: checkpoint files of the final model (after training 
has ended). This is what is used for both validation and inference.
- network_architecture.pdf (only if hiddenlayer is installed!): a pdf document with a figure of the network architecture in it.
- progress.png: A plot of the training (blue) and validation (red) loss during training. Also shows an approximation of 
the evlauation metric (green). This approximation is the average Dice score of the foreground classes. It should, 
however, only to be taken with a grain of salt because it is computed on randomly drawn patches from the validation 
data at the end of each epoch, and the aggregation of TP, FP and FN for the Dice computation treats the patches as if 
they all originate from the same volume ('global Dice'; we do not compute a Dice for each validation case and then 
average over all cases but pretend that there is only one validation case from which we sample patches). The reason for 
this is that the 'global Dice' is easy to compute during training and is still quite useful to evaluate whether a model 
is training at all or not. A proper validation is run at the end of the training.
- validation_raw: in this folder are the predicted validation cases after the training has finished. The summary.json 
contains the validation metrics (a mean over all cases is provided at the end of the file).

During training it is often useful to watch the progress. We therefore recommend that you have a look at the generated 
progress.png when running the first training. It will be updated after each epoch.

#### Multi GPU training
Yes. nnU-Net supports two different multi-GPU implementation: DataParallel (DP) and Distributed Data Parallel (DDP)
(but currently only on one host!). DDP is faster than DP and should be preferred if possible. However, if you did not 
install nnunet as a framework (meaning you used the `pip install nnunet` variant), DDP is not available. It requires a 
different way of calling the correct python script (see below) which we cannot support from our terminal commands.

Distributed training currently only works for the basic trainers (2D, 3D full resolution and 3D low resolution) and not 
for the second, high resolution U-Net of the cascade. The reason for this is that distributed training requires some 
changes to the network and loss function, requiring a new nnUNet trainer class. This is, as of now, simply not 
implemented for the cascade, but may be added in the future.

To run distributed training (DP), use the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2... nnUNet_train_DP CONFIGURATION nnUNetTrainerV2_DP TASK_NAME_OR_ID FOLD -gpus GPUS --dbs
```

Note that nnUNetTrainerV2 was replaced with nnUNetTrainerV2_DP. Just like before, CONFIGURATION can be 2d, 3d_lowres or 
3d_fullres. TASK_NAME_OR_ID refers to the task you would like to train and FOLD is the fold of the cross-validation. 
GPUS (integer value) specifies the number of GPUs you wish to train on. To specify which GPUs you want to use, please make use of the 
CUDA_VISIBLE_DEVICES envorinment variable to specify the GPU ids (specify as many as you configure with -gpus GPUS).
--dbs, if set, will distribute the batch size across GPUs. So if nnUNet configures a batch size of 2 and you run on 2 GPUs
, each GPU will run with a batch size of 1. If you omit --dbs, each GPU will run with the full batch size (2 for each GPU 
in this example for a total of batch size 4).

To run the DDP training you must have nnU-Net installed as a framework. Your current working directory must be the 
nnunet folder (the one that has the dataset_conversion, evaluation, experiment_planning, ... subfolders!). You can then run
the DDP training with the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2... python -m torch.distributed.launch --master_port=XXXX --nproc_per_node=Y run/run_training_DDP.py CONFIGURATION nnUNetTrainerV2_DDP TASK_NAME_OR_ID FOLD --dbs
```

XXXX must be an open port for process-process communication (something like 4321 will do on most systems). Y is the 
number of GPUs you wish to use. Remember that we do not (yet) support distributed training across compute nodes. This 
all happens on the same system. Again, you can use CUDA_VISIBLE_DEVICES=0,1,2 to control what GPUs are used.
If you run more than one DDP training on the same system (say you have 4 GPUs and you run two training with 2 GPUs each) 
you need to specify a different --master_port for each training!


### Identifying the best U-Net configuration(s)
Once all models are trained, use the following 
command to automatically determine what U-Net configuration(s) to use for test set prediction:

```bash
nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t XXX --allow_missing_pp --strict
```

(all 5 folds need to be completed for all specified configurations!)

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

Note that per default, inference will be done with all available folds. We very strongly recommend you use all 5 folds. 
Thus, all 5 folds must have been trained prior to running inference. The list of available folds nnU-Net found will be 
printed at the start of the inference.

## How to run inference with pretrained models
TODO, depends on how I upload the models

# Extending/Changing nnU-Net
Please refer to [this](documentation/extending_nnunet.md) guide.

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
  from the decathlon are examples for anisotropic data)

You do not have to run five-fold cross-validation all the time. If you want to test single model performance, use
 *all* for `FOLD` instead of a number. Note that this will then not give you an estimate of your performance on the 
 training set. You will also no tbe able to automatically identify which ensembling should be used and nnU-Net will 
 not be able to configure a postprocessing. 
 
CAREFUL: DO NOT use fold=all when you intend to run the cascade! You must run the cross-validation in 3d_lowres so 
that you get proper (=not overfitted) low resolution predictions.
 
#### Sharing Models
You can share trained models by simply sending the corresponding output folder from `RESULTS_FOLDER/nnUNet` to 
whoever you want share them with. The recipient can then use nnU-Net for inference with this model.

#### Can I run nnU-Net on smaller GPUs?
nnU-Net is guaranteed to run on GPUs with 11GB of memory. Many configurations may also run on 8 GB. If you wish to 
configure nnU-Net to use a different amount of GPU memory, simply adapt the reference value for the GPU memory estimation 
accordingly (with some slack because the whole thing is not an exact science!). For example, in 
[experiment_planner_baseline_3DUNet_v21_11GB.py](nnunet/experiment_planning/experiment_planner_baseline_3DUNet_v21_11GB.py) 
we provide an example that attempts to maximise the usage of GPU memory on 11GB as opposed to the default which leaves 
much more headroom). This is simply achieved by this line:

```python
ref = Generic_UNet.use_this_for_batch_size_computation_3D * 11 / 7.5
```

with 77.5 being what is currently used (approximately) and 11 being the target. Should you get CUDA out of memory 
issues, simply reduce the reference value. You should do this adaptation as part of a separate ExperimentPlanner class. 
Please read the instructions [here](documentation/extending_nnunet.md).

A 32 GB variant is also provided (ExperimentPlanner3D_v21_32GB). Note that increasing the GPU memory target while 
remaining on the same GPU will increase the computation time during training and thus the run time substantially! 

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
3d_lowres is created only if the patch size in 3d_fullres less than 1/8 of the voxels of the median shape of the data 
in 3d_fullres (for example Liver is about 512x512x512 and the patch size is 128x128x128, so that's 1/64 and thus 
3d_lowres is created). You can enforce the creation of 3d_lowres models for smaller datasets by changing the value of
`HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0` (located in experiment_planning.configuration).
    
