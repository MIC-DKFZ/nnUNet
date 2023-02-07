# Usage
To familiarize yourself with nnU-Net we recommend you have a look at the [Examples](#Examples) before you start with
your own dataset.

## How to run nnU-Net on a new dataset
Given some dataset, nnU-Net fully automatically configures an entire segmentation pipeline that matches its properties.
nnU-Net covers the entire pipeline, from preprocessing to model configuration, model training, postprocessing
all the way to ensembling. After running nnU-Net, the trained model(s) can be applied to the test cases for inference.

### Dataset conversion
nnU-Net expects datasets in a structured format. This format is inspired by the data structure of
the [Medical Segmentation Decthlon](http://medicaldecathlon.com/). Please read
[this](documentation/dataset_conversion.md) for information on how to convert datasets to be compatible with nnU-Net.

**Since version 2 we support multiple image file formats (.nii.gz, .png, .tif, ...)! Read the dataset conversion 
documentation to learn more!**

### Experiment planning and preprocessing
Given a new dataset, nnU-Net will extract a dataset fingerprint (a set of dataset-specific properties such as
image sizes, voxel spacings, intensity information etc). This information is used to design three U-Net configurations. 
Each of these pipelines operates on its own preprocessed version of the dataset.

The easiest way to run fingerprint extraction, experiment planning and preprocessing is to use:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_NAME_OR_ID --verify_dataset_integrity
```

Where `DATASET_NAME_OR_ID` is the dataset id or name (duh). So it's either an integer value (1, 2, 3, ...) or 
`Dataset001_ABDC`. We recommend `--verify_dataset_integrity` whenever it's the first time you run this command. This 
wiill check for some of the most common error sources!

You can also process several datasets at once by giving `-d 1 2 3 [...]`. If you already know what U-Net configuration 
you need you can also specify that with `-c 3d_fullres` (make sure to adapt -np in this case!). For more information 
about all the options availabel to you please run `nnUNetv2_plan_and_preprocess -h`.

nnUNetv2_plan_and_preprocess will create a new subfolder in your nnUNet_preprocessed folder named after the dataset. 
Once the command is completed there will be a dataset_fingerprint.json file as well as a plans for you to look at 
(in case you are interested!). There will also be subfolders containing the preprocessed data for your UNet configurations.

[Optional]
If you prefer to keep things separate, you can also use `nnUNetv2_extract_fingerprint`, `nnUNetv2_plan_experiment` 
and `nnUNetv2_preprocess` (in that order). 

### Model training
#### Overview
You pick which configurations (2D, 3D fullres, 3D cascade) should be trained! If you have no idea what performs best 
on your data, just run all of them and let nnU-Net identify the best one. It's up to you!

nnU-Net trains all configurations in a 5-fold cross-validation over the training cases. This is 1) needed so that 
nnU-Net can estimate the performance of each configuration and tell you which one should be used for your 
segmentation problem and 2) a natural way of obtaining a good model ensemble (average the output of these 5 models 
for prediction) to boost performance.

You can influence the splits nnU-Net uses for 5-fold cross-validation (see [here](XXXXX)). If you prefer to train a 
single model on all training cases, this is also possible (see below).

Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the U-Net
cascade is omitted because the patch size of the full resolution U-Net already covers a large part of the input images.


Training models is done with the `nnUNetv2_train` command. The general structure of the command is:
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```

UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres, 
3d_cascade_lowres). DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of 
the 5-fold-cross-validaton is trained.

nnU-Net stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a `-c` to the
training command.

IMPORTANT: If you plan to use `nnUNetv2_find_best_configuration` (see below) add the `--npz` flag. This makes 
nnU-Net save the softmax outputs during the final validation. They are needed for that. Exported softmax
predictions are very large and therefore can take up a lot of disk space.
If you ran initially without the `--npz` flag but now require the softmax predictions, simply rerun the validation with:
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD -val --npz
```

See `nnUNetv2_train -h` for additional options.

### 2D U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
```

### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
```

### 3D U-Net cascade
#### 3D low resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD [--npz]
```

#### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD [--npz]
```
**Note that the 3D full resolution U-Net of the cascade requires the five folds of the low resolution U-Net to be
completed!**

The trained models will be written to the nnUNet_results folder. Each training obtains an automatically generated
output folder name:

nnUNet_results/DatasetXXX_MYNAME/TRAINER_CLASS_NAME__PLANS_NAME__CONFIGURATION/FOLD

For Dataset002_Heart (from the MSD), for example, this looks like this:

    nnUNet_results/
    ├── Dataset002_Heart
        │── nnUNetTrainer__nnUNetPlans__2d
        │    ├── fold_0
        │    ├── fold_1
        │    ├── fold_2
        │    ├── fold_3
        │    ├── fold_4
        │    ├── dataset.json
        │    ├── dataset_fingerprint.json
        │    └── plans.json
        └── nnUNetTrainer__nnUNetPlans__3d_fullres
             ├── fold_0
             ├── fold_1
             ├── fold_2
             ├── fold_3
             ├── fold_4
             ├── dataset.json
             ├── dataset_fingerprint.json
             └── plans.json

Note that 3d_lowres and 3d_cascade_fullres do not exist here because this dataset did not trigger the cascade. In each
model training output folder (each of the fold_x folder), the following files will be created (only
shown for one folder above for brevity):
- debug.json: Contains a summary of blueprint and inferred parameters used for training this model. Not easy to read,
  but very useful for debugging ;-)
- checkpoint_best.pth: checkpoint files of the best model identified during training. Not used right now.
- checkpoint_final.pth: checkpoint files of the final model (after training
  has ended). This is what is used for both validation and inference.
- network_architecture.pdf (only if hiddenlayer is installed!): a pdf document with a figure of the network architecture in it.
- progress.png: A plot of the training (blue) and validation (red) loss during training. Also shows an approximation of
  the dice (green) as well as a moving average of it (dotted green line). This approximation is the average Dice score 
  of the foreground classes. **It needs to be taken with a big (!) 
  grain of salt** because it is computed on randomly drawn patches from the validation
  data at the end of each epoch, and the aggregation of TP, FP and FN for the Dice computation treats the patches as if
  they all originate from the same volume ('global Dice'; we do not compute a Dice for each validation case and then
  average over all cases but pretend that there is only one validation case from which we sample patches). The reason for
  this is that the 'global Dice' is easy to compute during training and is still quite useful to evaluate whether a model
  is training at all or not. A proper validation takes way too long to be done each epoch. It is run at the end of the training.
- validation_raw: in this folder are the predicted validation cases after the training has finished. The summary.json file in here
  contains the validation metrics (a mean over all cases is provided at the start of the file).

During training it is often useful to watch the progress. We therefore recommend that you have a look at the generated
progress.png when running the first training. It will be updated after each epoch.

Training times largely depend on the GPU. The smallest GPU we recommend for training is the Nvidia RTX 2080ti. With 
all network trainings take less than 2 days.

### Using multiple GPUs for training

If multiple GPUs are at your disposal, the best way of using them is to train multiple nnU-Net trainings at once, one 
on each GPU. This is because data parallelism never scales perfectly linearly, especially not with small networks such 
as the ones used by nnU-Net.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 [--npz] & # train on GPU 0
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train DATASET_NAME_OR_ID 2d 2 [--npz] & # train on GPU 0
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATASET_NAME_OR_ID 2d 3 [--npz] & # train on GPU 0
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train DATASET_NAME_OR_ID 2d 4 [--npz] & # train on GPU 0
wait
```

Important: The first time a training is run nnU-Net will extract the preprocessed data into a different file format. 
This operation must be completed before starting more than one fold of the same configuration! Wait with starting 
subsequent folds until the first training is using the GPU!

If you insist on running DDP multi-GPU training, we got you covered:

`nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] -num_gpus X`

Important when using `-num_gpus`:
1) If you train using, say, 2 GPUs but have more GPUs in the system you need to specify which GPUs should be used via 
CUDA_VISIBLE_DEVICES=0,1 (or whatever your ids are).
2) You cannot specify more GPUs than you have samples in your minibatches. If the batch size is 2, 2 GPUs is the maximum!
3) Make sure your batch size is divisible by the numbers of GPUs you use or you will not make good use of your hardware.

In contrast to the old nnU-Net, DDP is now completely hassle free. Enjoy!

### Automatically determine the best configuration


### Run inference
Remember that the data located in the input folder must have the file endings as the dataset you trained the model on 
and must adhere to the nnU-Net naming scheme for image files (see [dataset conversion](dataset_conversion.md)!)

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
modalities/channels as input in a specific order).

When using the pretrained models you must adhere to the license of the dataset they are trained on! If you run
`nnUNet_download_pretrained_model` you will find a link where you can find the license for each dataset.

## Examples

To get you started we compiled two simple to follow examples:
- run a training with the 3d full resolution U-Net on the Hippocampus dataset. See [here](documentation/training_example_Hippocampus.md).
- run inference with nnU-Net's pretrained models on the Prostate dataset. See [here](documentation/inference_example_Prostate.md).

Usability not good enough? Let us know!
