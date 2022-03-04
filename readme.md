**[2020_10_21] Update:** We now have documentation for [common questions](documentation/common_questions.md) and
[common issues](documentation/common_problems_and_solutions.md). We now also provide [reference epoch times for 
several datasets and tips on how to identify bottlenecks](documentation/expected_epoch_times.md).

Please read these documents before opening a new issue!

**################## Monjoy's System START #######**

** A. nnUNet data Preparation Instructions on Biowulf**

**Convert `nii` to `nii.gz`** This needs to be done on MATLAB. Follow the below instructions. 

1. Open [user@biowulf ~]$ sinteractive
2. [user@cn1234 ~]$ module load matlab
3. [user@cn1234 ~]$ matlab
4. gzip('*.nii') # all `nii` files belong to the folder will be converted into compress `nii.gz` file


** B. nnUNet Training and Inference Instructions on Biowulf:**

1. Open Biowulf
2. Go to folder "/data/saham2/Esophagus_Segmentation/nnU-Net_6jan2022/". "nnUNet" is the main file. 
3. Activate Conda using the following commands in the terminal

```source /data/saham2/conda/etc/profile.d/conda.sh```

```conda activate project2```

4. Set paths:

```export nnUNet_raw_data_base="/data/saham2/Esophagus_Segmentation/nnU-Net_6jan2022/nnUNet_raw_data_base/"```

```export nnUNet_preprocessed="/data/saham2/Esophagus_Segmentation/nnU-Net_6jan2022/nnUNet_preprocessed/"```

```export RESULTS_FOLDER="/data/saham2/Esophagus_Segmentation/nnU-Net_6jan2022/nnUNet_trained_models/"```

5. Follow instructions from this paper https://github.com/MIC-DKFZ/nnUNet/blob/master/readme.md#run-inference
6. libstdc++.so.6: version `CXXABI_1.3.9' not found. This problem can be solved by setting conda llib path like below. Type below code on the same terminal.

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/saham2/conda/lib``` [source: https://github.com/AllenDowney/ThinkStats2/issues/92].  

**MODEL TRAINING**

7. To train use following commands. Here for each batch training should be repeated. Batch should start with Zero (0). So 0, 1, 2, 3, 4 for 5 fold cross validation

**2D U-Net**

```nnUNet_train 2d nnUNetTrainerV2 Task055_SegTHOR 0 --npz```


#### 3D full resolution U-Net

```nnUNet_train 3d_fullres nnUNetTrainerV2 Task055_SegTHOR 0 --npz```

#### 3D cascade low resolution U-Net


```nnUNet_train 3d_lowres nnUNetTrainerV2 Task055_SegTHOR 0 --npz ```


**8. **Identifying the best U-Net configuration****

Once all models are trained, use the following command to automatically determine what U-Net configuration(s) to use for test set prediction:

```nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres  -t 055```

```--strict``` option is not required as this flag has been removed from the main script.
 see ```Output_of_bestModel.txt``` file for the output of above command
 
 **MODEL PREDICTION**
 
** **9. Run Inference****
Input folder need to be specified. Input folder will contain test images. In my case input folder path is given below:

```/data/saham2/Esophagus_Segmentation/nnU-Net_6jan2022/nnUNet_raw_data_base/nnUNet_raw_data/Task055_SegTHOR/imagesTs/```

**Full command-1:**

```nnUNet_predict -i /data/saham2/Esophagus_Segmentation/nnU-Net_6jan2022/nnUNet_raw_data_base/nnUNet_raw_data/Task055_SegTHOR/imagesTs/ -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task055_SegTHOR```

**Full command-2**

```nnUNet_predict -i /data/saham2/Esophagus_Segmentation/nnU-Net_6jan2022/nnUNet_raw_data_base/nnUNet_raw_data/Task055_SegTHOR/imagesTs/ -o OUTPUT_FOLDER_MODEL2 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_lowres -p nnUNetPlansv2.1 -t Task055_SegTHOR```

**Full command-3**

```nnUNet_ensemble -f OUTPUT_FOLDER_MODEL1 OUTPUT_FOLDER_MODEL2 -o OUTPUT_FOLDER -pp /data/saham2/Esophagus_Segmentation/nnU-Net_6jan2022/nnUNet_trained_models/nnUNet/ensembles/Task055_SegTHOR/ensemble_3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1--3d_lowres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json```


**################## Monjoy's System END #######**



# nnU-Net

In 3D biomedical image segmentation, dataset properties like imaging modality, image sizes, voxel spacings, class 
ratios etc vary drastically.
For example, images in the [Liver and Liver Tumor Segmentation Challenge dataset](https://competitions.codalab.org/competitions/17094) 
are computed tomography (CT) scans, about 512x512x512 voxels large, have isotropic voxel spacings and their 
intensity values are quantitative (Hounsfield Units).
The [Automated Cardiac Diagnosis Challenge dataset](https://acdc.creatis.insa-lyon.fr/) on the other hand shows cardiac 
structures in cine MRI with a typical image shape of 10x320x320 voxels, highly anisotropic voxel spacings and 
qualitative intensity values. In addition, the ACDC dataset suffers from slice misalignments and a heterogeneity of 
out-of-plane spacings which can cause severe interpolation artifacts if not handled properly. 

In current research practice, segmentation pipelines are designed manually and with one specific dataset in mind. 
Hereby, many pipeline settings depend directly or indirectly on the properties of the dataset 
and display a complex co-dependence: image size, for example, affects the patch size, which in 
turn affects the required receptive field of the network, a factor that itself influences several other 
hyperparameters in the pipeline. As a result, pipelines that were developed on one (type of) dataset are inherently 
incomaptible with other datasets in the domain.

**nnU-Net is the first segmentation method that is designed to deal with the dataset diversity found in the domain. It 
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


    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
    for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

Please also cite this paper if you are using nnU-Net for your research!


# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  * [How to run nnU-Net on a new dataset](#how-to-run-nnu-net-on-a-new-dataset)
    + [Dataset conversion](#dataset-conversion)
    + [Experiment planning and preprocessing](#experiment-planning-and-preprocessing)
    + [Model training](#model-training)
      - [2D U-Net](#2d-u-net)
      - [3D full resolution U-Net](#3d-full-resolution-u-net)
      - [3D U-Net cascade](#3d-u-net-cascade)
        * [3D low resolution U-Net](#3d-low-resolution-u-net)
        * [3D full resolution U-Net](#3d-full-resolution-u-net-1)
      - [Multi GPU training](#multi-gpu-training)
    + [Identifying the best U-Net configuration](#identifying-the-best-u-net-configuration)
    + [Run inference](#run-inference)
  * [How to run inference with pretrained models](#how-to-run-inference-with-pretrained-models)
  * [Examples](#examples)
- [Extending/Changing nnU-Net](#extending-or-changing-nnu-net)
- [Information on run time and potential performance bottlenecks.](#information-on-run-time-and-potential-performance-bottlenecks)
- [Common questions and issues](#common-questions-and-issues)


# Installation
nnU-Net has been tested on Linux (Ubuntu 16, 18 and 20; centOS, RHEL). We do not provide support for other operating 
systems.

nnU-Net requires a GPU! For inference, the GPU should have 4 GB of VRAM. For training nnU-Net models the GPU should have at 
least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080 or RTX 3090). Due to the use of automated mixed 
precision, fastest training times are achieved with the Volta architecture (Titan V, V100 GPUs) when installing pytorch 
the easy way. Since pytorch comes with cuDNN 7.6.5 and tensor core acceleration on Turing GPUs is not supported for 3D 
convolutions in this version, you will not get the best training speeds on Turing GPUs. You can remedy that by compiling pytorch from source 
(see [here](https://github.com/pytorch/pytorch#from-source)) using cuDNN 8.0.2 or newer. This will unlock Turing GPUs 
(RTX 2080ti, RTX 6000) for automated mixed precision training with 3D convolutions and make the training blistering 
fast as well. Note that future versions of pytorch may include cuDNN 8.0.2 or newer by default and 
compiling from source will not be necessary.
We don't know the speed of Ampere GPUs with vanilla vs self-compiled pytorch yet - this section will be updated as 
soon as we know.

For training, we recommend a strong CPU to go along with the GPU. At least 6 CPU cores (12 threads) are recommended. CPU 
requirements are mostly related to data augmentation and scale with the number of input channels. They are thus higher 
for datasets like BraTS which use 4 image modalities and lower for datasets like LiTS which only uses CT images.

We very strongly recommend you install nnU-Net in a virtual environment. 
[Here is a quick how-to for Ubuntu.](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/)
If you choose to compile pytorch from source, you will need to use conda instead of pip. In that case, please set the 
environment variable OMP_NUM_THREADS=1 (preferably in your bashrc using `export OMP_NUM_THREADS=1`). This is important!

Python 2 is deprecated and not supported. Please make sure you are using Python 3.

1) Install [PyTorch](https://pytorch.org/get-started/locally/). You need at least version 1.6
2) Install nnU-Net depending on your use case:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running **inference with pretrained models**:
      
        ```pip install nnunet```
    
    2) For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
3) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to 
set a few of environment variables. Please follow the instructions [here](documentation/setting_up_paths.md).
4) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate 
plots of the network topologies it generates (see [Model training](#model-training)). To install hiddenlayer, 
run the following commands:
    ```bash
    pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net 
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNet_` for 
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this 
environment must be activated when executing the commands.

All nnU-Net commands have a `-h` option which gives information on how to use them.

A typical installation of nnU-Net can be completed in less than 5 minutes. If pytorch needs to be compiled from source 
(which is what we currently recommend when using Turing GPUs), this can extend to more than an hour.

# Usage
To familiarize yourself with nnU-Net we recommend you have a look at the [Examples](#Examples) before you start with 
your own dataset.

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

Extraction of the dataset fingerprint can take from a couple of seconds to several minutes depending on the properties 
of the segmentation task. Pipeline configuration given the extracted finger print is nearly instantaneous (couple 
of seconds). Preprocessing depends on image size and how powerful the CPU is. It can take between seconds and several 
tens of minutes.

### Model training
nnU-Net trains all U-Net configurations in a 5-fold cross-validation. This enables nnU-Net to determine the 
postprocessing and ensembling (see next step) on the training dataset. Per default, all U-Net configurations need to 
be run on a given dataset. There are, however situations in which only some configurations (and maybe even without 
running the cross-validation) are desired. See [FAQ](documentation/common_questions.md) for more information.

Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the U-Net 
cascade is omitted because the patch size of the full resolution U-Net already covers a large part of the input images.

Training models is done with the `nnUNet_train` command. The general structure of the command is:
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD  --npz (additional options)
```

CONFIGURATION is a string that identifies the requested U-Net configuration. TRAINER_CLASS_NAME is the name of the 
model trainer. If you implement custom trainers (nnU-Net as a framework) you can specify your custom trainer here.
TASK_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of the 5-fold-cross-validaton 
is trained.

nnU-Net stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a `-c` to the 
training command.

IMPORTANT: `--npz` makes the models save the softmax outputs during the final validation. It should only be used for trainings 
where you plan to run `nnUNet_find_best_configuration` afterwards 
(this is nnU-Nets automated selection of the best performing (ensemble of) configuration(s), see below). If you are developing new 
trainer classes you may not need the softmax predictions and should therefore omit the `--npz` flag. Exported softmax 
predictions are very large and therefore can take up a lot of disk space.
If you ran initially without the `--npz` flag but now require the softmax predictions, simply run 
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD -val --npz
```
to generate them. This will only rerun the validation, not the training.

See `nnUNet_train -h` for additional options.

#### 2D U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 2d nnUNetTrainerV2 TaskXXX_MYTASK FOLD --npz
```

#### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 TaskXXX_MYTASK FOLD --npz
```

#### 3D U-Net cascade
##### 3D low resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_lowres nnUNetTrainerV2 TaskXXX_MYTASK FOLD --npz
```

##### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes TaskXXX_MYTASK FOLD --npz
```

Note that the 3D full resolution U-Net of the cascade requires the five folds of the low resolution U-Net to be 
completed beforehand!

The trained models will be written to the RESULTS_FOLDER/nnUNet folder. Each training obtains an automatically generated 
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

Training times largely depend on the GPU. The smallest GPU we recommend for training is the Nvidia RTX 2080ti. With 
this GPU (and pytorch compiled with cuDNN 8.0.2), all network trainings take less than 2 days.

#### Multi GPU training

**Multi GPU training is experimental and NOT RECOMMENDED!**

nnU-Net supports two different multi-GPU implementation: DataParallel (DP) and Distributed Data Parallel (DDP)
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

*IMPORTANT!*
Multi-GPU training results in models that cannot be used for inference easily (as said above, all of this is experimental ;-) ).
After finishing the training of all folds, run `nnUNet_change_trainer_class` on the folder where the trained model is 
(see `nnUNet_change_trainer_class -h` for instructions). After that you can run inference.

### Identifying the best U-Net configuration
Once all models are trained, use the following 
command to automatically determine what U-Net configuration(s) to use for test set prediction:

```bash
nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t XXX --strict
```

(all 5 folds need to be completed for all specified configurations!)

On datasets for which the cascade was not configured, use `-m 2d 3d_fullres` instead. If you wish to only explore some 
subset of the configurations, you can specify that with the `-m` command. We recommend setting the 
`--strict` (crash if one of the requested configurations is 
missing) flag. Additional options are available (use `-h` for help).

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

Trained models for all challenges we participated in are publicly available. They can be downloaded and installed 
directly with nnU-Net. Note that downloading a pretrained model will overwrite other models that were trained with 
exactly the same configuration (2d, 3d_fullres, ...), trainer (nnUNetTrainerV2) and plans.

To obtain a list of available models, as well as a short description, run

```bash
nnUNet_print_available_pretrained_models
```

You can then download models by specifying their task name. For the Liver and Liver Tumor Segmentation Challenge, 
for example, this would be:

```bash
nnUNet_download_pretrained_model Task029_LiTS
```
After downloading is complete, you can use this model to run [inference](#run-inference). Keep in mind that each of 
these models has specific data requirements (Task029_LiTS runs on abdominal CT scans, others require several image 
modalities as input in a specific order).

When using the pretrained models you must adhere to the license of the dataset they are trained on! If you run 
`nnUNet_download_pretrained_model` you will find a link where you can find the license for each dataset.

## Examples

To get you started we compiled two simple to follow examples:
- run a training with the 3d full resolution U-Net on the Hippocampus dataset. See [here](documentation/training_example_Hippocampus.md).
- run inference with nnU-Net's pretrained models on the Prostate dataset. See [here](documentation/inference_example_Prostate.md).

Usability not good enough? Let us know!

# Extending or Changing nnU-Net
Please refer to [this](documentation/extending_nnunet.md) guide.

# Information on run time and potential performance bottlenecks.

We have compiled a list of expected epoch times on standardized datasets across many different GPUs. You can use them 
to verify that your system is performing as expected. There are also tips on how to identify bottlenecks and what 
to do about them.

Click [here](documentation/expected_epoch_times.md).

# Common questions and issues

We have collected solutions to common [questions](documentation/common_questions.md) and 
[problems](documentation/common_problems_and_solutions.md). Please consult these documents before you open a new issue.

--------------------

<img src="HIP_Logo.png" width="512px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of the [Helmholtz Imaging Platform](http://helmholtz-imaging.de).
