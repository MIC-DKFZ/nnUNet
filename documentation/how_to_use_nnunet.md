## How to run nnU-Net on a new dataset
Given some dataset, nnU-Net fully automatically configures an entire segmentation pipeline that matches its properties.
nnU-Net covers the entire pipeline, from preprocessing to model configuration, model training, postprocessing
all the way to ensembling. After running nnU-Net, the trained model(s) can be applied to the test cases for inference.

### Dataset Format
nnU-Net expects datasets in a structured format. This format is inspired by the data structure of
the [Medical Segmentation Decthlon](http://medicaldecathlon.com/). Please read
[this](documentation/dataset_format.md) for information on how to set up datasets to be compatible with nnU-Net.

**Since version 2 we support multiple image file formats (.nii.gz, .png, .tif, ...)! Read the dataset_format 
documentation to learn more!**

**Datasets from nnU-Net v1 can be converted to V2 by running `nnUNetv2_convert_old_nnUNet_dataset INPUT_FOLDER 
OUTPUT_DATASET_NAME`.** Remember that v2 calls datasets DatasetXXX_Name (not Task) where XXX is a 3-digit number.
Please provide the **path** to the old task, not just the Task name. nnU-Net V2 doesn't know where v1 tasks were!

### Experiment planning and preprocessing
Given a new dataset, nnU-Net will extract a dataset fingerprint (a set of dataset-specific properties such as
image sizes, voxel spacings, intensity information etc). This information is used to design three U-Net configurations. 
Each of these pipelines operates on its own preprocessed version of the dataset.

The easiest way to run fingerprint extraction, experiment planning and preprocessing is to use:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Where `DATASET_ID` is the dataset id (duh). We recommend `--verify_dataset_integrity` whenever it's the first time 
you run this command. This will check for some of the most common error sources!

You can also process several datasets at once by giving `-d 1 2 3 [...]`. If you already know what U-Net configuration 
you need you can also specify that with `-c 3d_fullres` (make sure to adapt -np in this case!). For more information 
about all the options available to you please run `nnUNetv2_plan_and_preprocess -h`.

nnUNetv2_plan_and_preprocess will create a new subfolder in your nnUNet_preprocessed folder named after the dataset. 
Once the command is completed there will be a dataset_fingerprint.json file as well as a nnUNetPlans.json file for you to look at 
(in case you are interested!). There will also be subfolders containing the preprocessed data for your UNet configurations.

[Optional]
If you prefer to keep things separate, you can also use `nnUNetv2_extract_fingerprint`, `nnUNetv2_plan_experiment` 
and `nnUNetv2_preprocess` (in that order). 

### Model training
#### Overview
You pick which configurations (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres) should be trained! If you have no idea 
what performs best on your data, just run all of them and let nnU-Net identify the best one. It's up to you!

nnU-Net trains all configurations in a 5-fold cross-validation over the training cases. This is 1) needed so that 
nnU-Net can estimate the performance of each configuration and tell you which one should be used for your 
segmentation problem and 2) a natural way of obtaining a good model ensemble (average the output of these 5 models 
for prediction) to boost performance.

You can influence the splits nnU-Net uses for 5-fold cross-validation (see [here](manual_data_splits.md)). If you 
prefer to train a single model on all training cases, this is also possible (see below).

Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the U-Net
cascade (and with it the 3d_lowres configuration) is omitted because the patch size of the full resolution U-Net 
already covers a large part of the input images.

Training models is done with the `nnUNetv2_train` command. The general structure of the command is:
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```

UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres, 
3d_cascade_lowres). DATASET_NAME_OR_ID specifies what dataset should be trained on and FOLD specifies which fold of 
the 5-fold-cross-validation is trained.

nnU-Net stores a checkpoint every 50 epochs. If you need to continue a previous training, just add a `--c` to the
training command.

IMPORTANT: If you plan to use `nnUNetv2_find_best_configuration` (see below) add the `--npz` flag. This makes 
nnU-Net save the softmax outputs during the final validation. They are needed for that. Exported softmax
predictions are very large and therefore can take up a lot of disk space, which is why this is not enabled by default.
If you ran initially without the `--npz` flag but now require the softmax predictions, simply rerun the validation with:
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz
```

You can specify the device nnU-net should use by using `-device DEVICE`. DEVICE can only be cpu, cuda or mps. If 
you have multiple GPUs, please select the gpu id using `CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...]` (requires device to be cuda).

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
model training output folder (each of the fold_x folder), the following files will be created:
- debug.json: Contains a summary of blueprint and inferred parameters used for training this model as well as a 
bunch of additional stuff. Not easy to read, but very useful for debugging ;-)
- checkpoint_best.pth: checkpoint files of the best model identified during training. Not used right now unless you 
explicitly tell nnU-Net to use it.
- checkpoint_final.pth: checkpoint file of the final model (after training has ended). This is what is used for both 
validation and inference.
- network_architecture.pdf (only if hiddenlayer is installed!): a pdf document with a figure of the network architecture in it.
- progress.png: Shows losses, pseudo dice, learning rate and epoch times ofer the course of the training. At the top is 
a plot of the training (blue) and validation (red) loss during training. Also shows an approximation of
  the dice (green) as well as a moving average of it (dotted green line). This approximation is the average Dice score 
  of the foreground classes. **It needs to be taken with a big (!) 
  grain of salt** because it is computed on randomly drawn patches from the validation
  data at the end of each epoch, and the aggregation of TP, FP and FN for the Dice computation treats the patches as if
  they all originate from the same volume ('global Dice'; we do not compute a Dice for each validation case and then
  average over all cases but pretend that there is only one validation case from which we sample patches). The reason for
  this is that the 'global Dice' is easy to compute during training and is still quite useful to evaluate whether a model
  is training at all or not. A proper validation takes way too long to be done each epoch. It is run at the end of the training.
- validation_raw: in this folder are the predicted validation cases after the training has finished. The summary.json file in here
  contains the validation metrics (a mean over all cases is provided at the start of the file). If `--npz` was set then 
the compressed softmax outputs (saved as .npz files) are in here as well. 

During training it is often useful to watch the progress. We therefore recommend that you have a look at the generated
progress.png when running the first training. It will be updated after each epoch.

Training times largely depend on the GPU. The smallest GPU we recommend for training is the Nvidia RTX 2080ti. With 
that all network trainings take less than 2 days. Refer to our [benchmarks](benchmarking.md) to see if your system is 
performing as expected.

### Using multiple GPUs for training

If multiple GPUs are at your disposal, the best way of using them is to train multiple nnU-Net trainings at once, one 
on each GPU. This is because data parallelism never scales perfectly linearly, especially not with small networks such 
as the ones used by nnU-Net.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] & # train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 [--npz] & # train on GPU 1
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train DATASET_NAME_OR_ID 2d 2 [--npz] & # train on GPU 2
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATASET_NAME_OR_ID 2d 3 [--npz] & # train on GPU 3
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train DATASET_NAME_OR_ID 2d 4 [--npz] & # train on GPU 4
...
wait
```

**Important: The first time a training is run nnU-Net will extract the preprocessed data into uncompressed numpy 
arrays for speed reasons! This operation must be completed before starting more than one training of the same 
configuration! Wait with starting subsequent folds until the first training is using the GPU! Depending on the 
dataset size and your System this should oly take a couple of minutes at most.**

If you insist on running DDP multi-GPU training, we got you covered:

`nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] -num_gpus X`

Again, note that this will be slower than running separate training on separate GPUs. DDP only makes sense if you have 
manually interfered with the nnU-Net configuration and are training larger models with larger patch and/or batch sizes!

Important when using `-num_gpus`:
1) If you train using, say, 2 GPUs but have more GPUs in the system you need to specify which GPUs should be used via 
CUDA_VISIBLE_DEVICES=0,1 (or whatever your ids are).
2) You cannot specify more GPUs than you have samples in your minibatches. If the batch size is 2, 2 GPUs is the maximum!
3) Make sure your batch size is divisible by the numbers of GPUs you use or you will not make good use of your hardware.

In contrast to the old nnU-Net, DDP is now completely hassle free. Enjoy!

### Automatically determine the best configuration
Once the desired configurations were trained (full cross-validation) you can tell nnU-Net to automatically identify 
the best combination for you:

```commandline
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
```

`CONFIGURATIONS` hereby is the list of configurations you would like to explore. Per default, ensembling is enabled 
meaning that nnU-Net will generate all possible combinations of ensembles (2 configurations per ensemble). This requires 
the .npz files containing the predicted probabilities of the validation set to be present (use `nnUNetv2_train` with 
`--npz` flag, see above). You can disable ensembling by setting the `--disable_ensembling` flag.

See `nnUNetv2_find_best_configuration -h` for more options.

nnUNetv2_find_best_configuration will also automatically determine the postprocessing that should be used. 
Postprocessing in nnU-Net only considers the removal of all but the largest component in the prediction (once for 
foreground vs background and once for each label/region).

Once completed, the command will print to your console exactly what commands you need to run to make predictions. It 
will also create two files in the `nnUNet_results/DATASET_NAME` folder for you to inspect: 
- `inference_instructions.txt` again contains the exact commands you need to use for predictions
- `inference_information.json` can be inspected to see the performance of all configurations and ensembles, as well 
as the effect of the postprocessing plus some debug information. 

### Run inference
Remember that the data located in the input folder must have the file endings as the dataset you trained the model on 
and must adhere to the nnU-Net naming scheme for image files (see [dataset format](dataset_format.md) and 
[inference data format](dataset_format_inference.md)!)

`nnUNetv2_find_best_configuration` (see above) will print a string to the terminal with the inference commands you need to use.
The easiest way to run inference is to simply use these commands.

If you wish to manually specify the configuration(s) used for inference, use the following commands:

#### Run prediction
For each of the desired configurations, run:
```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

Only specify `--save_probabilities` if you intend to use ensembling. `--save_probabilities` will make the command save the predicted
probabilities alongside of the predicted segmentation masks requiring a lot of disk space.

Please select a separate `OUTPUT_FOLDER` for each configuration!

Note that per default, inference will be done with all 5 folds from the cross-validation as an ensemble. We very 
strongly recommend you use all 5 folds. Thus, all 5 folds must have been trained prior to running inference. 

If you wish to make predictions with a single model, train the `all` fold and specify it in `nnUNetv2_predict`
with `-f all`

#### Ensembling multiple configurations
If you wish to ensemble multiple predictions (typically form different configurations), you can do so with the following command:
```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -np NUM_PROCESSES
```

You can specify an arbitrary number of folders, but remember that each folder needs to contain npz files that were
generated by `nnUNetv2_predict`. Again, `nnUNetv2_ensemble -h` will tell you more about additional options.

#### Apply postprocessing
Finally, apply the previously determined postprocessing to the (ensembled) predictions: 

```commandline
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```

`nnUNetv2_find_best_configuration` (or its generated `inference_instructions.txt` file) will tell you where to find 
the postprocessing file. If not you can just look for it in your results folder (it's creatively named 
`postprocessing.pkl`). If your source folder is from an ensemble, you also need to specify a `-plans_json` file and 
a `-dataset_json` file that should be used (for single configuration predictions these are automatically copied 
from the respective training). You can pick these files from any of the ensemble members.


## How to run inference with pretrained models
See [here](run_inference_with_pretrained_models.md)

[//]: # (## Examples)

[//]: # ()
[//]: # (To get you started we compiled two simple to follow examples:)

[//]: # (- run a training with the 3d full resolution U-Net on the Hippocampus dataset. See [here]&#40;documentation/training_example_Hippocampus.md&#41;.)

[//]: # (- run inference with nnU-Net's pretrained models on the Prostate dataset. See [here]&#40;documentation/inference_example_Prostate.md&#41;.)

[//]: # ()
[//]: # (Usability not good enough? Let us know!)
