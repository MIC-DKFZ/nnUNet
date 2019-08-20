# Introduction 

nnU-Net is a framework designed for medical image segmentation. Given a new dataset (that includes training cases) nnU-Net
will automatically take care of the entire experimental pipeline.
Unlike other segmentation methods published recently, nnU-Net does not use complicated architectural modifications and 
instead revolves around the popular U-Net architecture. Still, nnU-Net outperforms many other methods and has been 
shown to produce segmentations that are on par with or even exceed the state-of-the art across six well-known medical 
segmentation challenges.

For more information about nnU-Net, please read the following paper:

`Isensee, Fabian, et al. "nnU-Net: Breaking the Spell on Successful Medical Image Segmentation." arXiv preprint arXiv:1904.08128 (2019).`

Please also cite this paper if you are using nnU-Net for your research!

Please note that so far nnU-Net has only been used internally. Tha vast majority of nnU-Net was developed in the context of 
the Medical Segmentation Decathlon and was thus created with a very tight time schedule, so expect things to be a 
little messy the deeper you dig.  

This repository is still work in progress. Things may break. If that is the case, please let us know.

# Installation 
nnU-Net is only tested on Linux (Ubuntu). It may work on other operating systems as well but we do not guarantee it will.

Installation instructions
1) Install PyTorch (https://pytorch.org/get-started/locally/)
2) Clone this repository `git clone https://github.com/MIC-DKFZ/nnUNet.git`
3) Go into the repository (`cd nnUNet` on linux)
4) Install with `pip install -r requirements.txt` followed by `pip install -e .`
5) There is an issue with numpy and threading in python which results in too high CPU usage when using multiprocessing. 
We strongly recommend you set OMP_NUM_THREADS=1 in your bashrc OR start all python processes 
with `OMP_NUM_THREADS=1 python ...`

# Getting Started 
All the commands in this section assume that you are in a terminal and your working directory is the nnU-Net folder 
(the one that has all the subfolders like `dataset_conversion`, `evaluation`, ...)

## Set paths 
nnU-Net needs to know where you will store raw data, want it to store preprocessed data and trained models. Have a 
look at the file `paths.py` and adapt it to your system.

## Preparing Datasets 
nnU-Net was initially developed as our participation to the Medical Segmentation Decathlon <sup>1</sup>. It therefore
 relies on the dataset to be in the same format as this challenge uses. Please refer to the readme.md in the 
 `dataset_conversion` subfolder for detailed information. Examples are also provided there. You will need to 
 convert your dataset into this format before you can continue.
 
Place your dataset either in the `raw_dataset_dir` or `splitted_4d_output_dir`, as specified in `paths.py` (depending on how you prepared it, again 
see the readme in `dataset_conversion`). Give 
it a name like: `TaskXX_MY_DATASET` (where XX is some number) to be consistent with the naming scheme of the Medical 
Segmentation Decathlon.

## Experiment Planning and Preprocessing 
This is where the magic happens. nnU-Net can now analyze your dataset and determine how to train its 
U-Net models. To run experiment planning and preprocessing for your dataset, execute the following command:

`python experiment_planning/plan_and_preprocess_task.py -t TaskXX_MY_DATASET -pl Y -pf Z`

here `TaskXX_MY_DATASET` specifies the task (your dataset) and `-pl`/`-pf` determines how many processes will be used for 
datatset analysis and preprocessing (see `python experiment_planning/plan_and_preprocess_task.py -h` for more details). Generally you want this number to be as high as you have CPU cores, unless you 
run into memory problems (beware of datasets such as LiTS!)

Running this command will to several things:
1) If you stored your data as 4D nifti the data will be split into a sequence of 3d niftis. Back when I started 
SimpleITK did not support 4D niftis. This was simply done out of necessity.
2) Images are cropped to the nonzero region. This does nothing for most datasets. Most brain datasets however are brain 
extracted, meaning that the brain is surrounded by zeros. There is no need to push all these zeros through the GPUs so 
we simply save a little bit of time doing this. Cropped data is stored in the `cropped_output_dir` (`paths.py`).
3) The data is analyzed and information about spacing, intensity distributions and shapes are determined
4) nnU-Net configures the U-Net architectures based on that information. All U-Nets are configured to optimally use 
**12GB Nvidia TitanX** GPUs. There is currently no way of adapting to smaller or larger GPUs.
5) nnU-Net runs the preprocessing and saves the preprocessed data in `preprocessing_output_dir`.

I strongly recommend you set `preprocessing_output_dir` on a SSD. HDDs are typically too slow for data loading. 

## Training Models 
There is an issue with numpy and threading in python which results in too high CPU usage when using multiprocessing. 
We strongly recommend you set OMP_NUM_THREADS=1 in your bashrc OR start all python processes 
with `OMP_NUM_THREADS=1 python ...`

The following pipeline describes what we ran for all the challenge submissions. If you are not interested in getting 
every last bit of performance we recommend you also look at the Recommendations section.

nnU-Net uses three different U-Net models and can automatically choose which (of what ensemble) of them to use. The 
default setting is to train each of these models in a five-fold cross-validation.

Trained models are stored in `network_training_output_dir` (specified in `paths.py`).

### 2D U-Net 
For `FOLD` in [0, 4], run:

`python run/run_training.py 2d nnUNetTrainer TaskXX_MY_DATASET FOLD --ndet`

### 3D U-Net (full resolution)
For `FOLD` in [0, 4], run:

`python run/run_training.py 3d_fullres nnUNetTrainer TaskXX_MY_DATASET FOLD --ndet`

### 3D U-Net Cascade 
The 3D U-Net cascade only applies to datasets where the patch size possible in the 'fullres' setting is too small 
relative to the size of the image data. If the cascade was configured you can run it as follows, otherwise this step 
can be skipped.

For `FOLD` in [0, 4], run:

`python run/run_training.py 3d_lowres nnUNetTrainer TaskXX_MY_DATASET FOLD --ndet`

After validation these models will automatically also predict the segmentations for the next stage of the cascade and 
save them in the correct spacing.

Then run
For `FOLD` in [0, 4], run:

`python run/run_training.py 3d_cascade_fullres nnUNetTrainerCascadeFullRes TaskXX_MY_DATASET FOLD --ndet`

## Ensembling 
Once everything that needs to be trained has been trained nnU-Net can ensemble the cross-validation results to figure 
out what the best combination of models is:

`python evaluation/model_selection/figure_out_what_to_submit.py -t XX`

where `XX` is the taskID you set for your dataset. This will generate as csv file in `network_training_output_dir` 
with the results.

You can also give a list of task ids to summarize several datastes at once.

## Inference 
You can use trained models to predict test data. In order to be able to do so the test data must be provided in the 
same format as the training data. Specifically, the data must be splitted in 3D niftis, so if you have more than one 
modality the files must be named like this (same format as nnUNet_raw_splitted! see readme in dataset_conversion folder):

```
CaseIdentifier1_0000.nii.gz, CaseIdnetifier1_0001.nii.gz, ...
CaseIdentifier2_0000.nii.gz, CaseIdnetifier2_0001.nii.gz, ...
...
```

To run inference for 3D U-Net model, use the following script:

`python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t TaskXX_MY_DATASET -tr nnUNetTrainer -m 3d_fullres`

If you wish to use the 2D U-Nets, you can set `-m 2d` instead of `3d_fullres`.

To run inference with the cascade, run the following two commands:

`python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER_LOWRES -t TaskXX_MY_DATASET -tr nnUNetTrainer -m 3d_lowres`

`python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER_CASCADE -t TaskXX_MY_DATASET -tr 
nnUNetTrainerCascadeFullRes -m 3d_cascade_fullres -l OUTPUT_FOLDER_LOWRES`

here we first predict the low resolution segmentations and then use them for the second stage of the cascade.

There are a lot more flags you can set for inference. Please consult the help of predict_simple.py for more information.

### Ensembling test cases 
Per default nnU-Net uses the five models obtained from cross-validation as an ensemble. How to ensemble different U-Net Models 
(for example 2D and 3D U-Net) is explained below.

If you wish to ensemble test cases, run all inference commands with the `-z` argument. This will tell nnU-Net to save the 
softmax probabilities as well. They are needed for ensembling.

You can then ensemble the predictions of two output folders with the following command:

`python inference/ensemble_predictions.py -f FOLDER1 FODLER2 ... -o OUTPUT_FOLDER`

This will ensemble the predictions located in `FODLER1, FOLDER2, ...` and write them into `OUTPUT_FOLDER`


## Tips and Tricks
The model training pipeline above is for challenge participations. Depending on your task you may not want to train all 
U-Net models and you may also not want to run a cross-validation all the time.

#### I don't want to train all these models. Which one is the best?
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
 
#### Manual Splitting of Data
The cross-validation in nnU-Net splits on a per-case basis. This may sometimes not be desired, for example because 
several training cases may be the same patient (different time steps or annotators). If this is the case, then you need to
manually create a split file. To do this, first let nnU-Net create the default split file. Run one of the network 
trainings (any of them works fine for this) and abort after the first epoch. nnU-Net will have created a split file automatically:
`preprocessing_output_dir/TaskXX_MY_DATASET/splits_final.pkl`. This file contains a list (length 5, one entry per fold). 
Each entry in the list is a dictionary with keys 'train' and 'val' pointing to the patientIDs assigned to the sets. 
To use your own splits in nnU-Net, you need to edit these entries to what you want them to be and save it back to the 
splits_final.pkl file. Use load_pickle and save_pickle from batchgenerators.utilities.file_and_folder_operations for convenience.

#### Sharing Models
You can share trained models by simply sending the corresponding output folder from `network_training_output_dir` to 
whoever you want share them with. The recipient can then use nnU-Net for inference with this model.

## FAQ
1) ##### Can I run nnU-Net on smaller GPUs?

    You can run nnU-Net in fp16 by specifying `--fp16` as additional option when launching trainings. This will reduce 
    the amount of GPU memory needed to ~9 GB and allow to run everything on 11GB cards as well. You can also manually 
    edit the plans.pkl files (that are located in the subfolders of preprocessed_output_dir) to make nnU-net use less 
    feature maps. This can however have an impact on segmentation performance

2) ##### I get the error `seg from prev stage missing` when running the cascade

    You need to run all five folds of `3d_lowres`. Segmentations of the previous stage can only be generated from the 
    validation set, otherwise we would overfit.
   
3) ##### Why am I getting `RuntimeError: CUDA error: device-side assert triggered`?

    This error often goes along with something like `void THCudaTensor_scatterFillKernel(TensorInfo<Real, IndexType>, 
    TensorInfo<long, IndexType>, Real, int, IndexType) [with IndexType = unsigned int, Real = float, Dims = -1]: 
    block: [4770,0,0], thread: [374,0,0] Assertion indexValue >= 0 && indexValue < tensor.sizes[dim] failed.`.
    
    This means that your dataset contains unexpected values in the segmentations. nnU-Net expects all labels to be 
    consecutive integers. So if your dataset has 4 classes (background and three foregound labels), then the labels 
    must be 0, 1, 2, 3 (where 0 must be background!). There cannot be any other values in the ground truth segmentations. 
    
## Extending nnU-Net
nnU-Net was developed in a very short amount of time and has not been planned thoroughly form the start (this is not 
really possible for such a project). As such it is quite convoluted and complex, maybe unnessearily so. If you wish to 
extend nnU-Net, ask questions if something is not clear. But please also keep in mind that this 
software is provided 'as is' and the amount of support we can give is limited :-)

## Changelog
nothing so far




<sup>1</sup>http://medicaldecathlon.com/