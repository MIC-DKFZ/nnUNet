# FAQ

## Creating and managing data splits

At the start of each training, nnU-Net will check whether the splits_final.pkl file is present in the directory where 
the preprocessed data of the requested dataset is located. If the file is not present, nnU-Net will create its own 
split: a five-fold cross-validation using all the available training cases. nnU-Net needs this five-fold 
cross-validation to be able to determine the postprocessing and to run model/ensemble selection.

There are however situations in which you may want to create your own split, for example
- in datasets like ACDC where several training cases are connected (there are two time steps for each patient) you 
may need to manually create splits to ensure proper stratification.
- cases are annotated by multiple annotators and you would like to use the annotations as separate training examples
- if you are running experiments with a domain transfer, you might want to train only on cases from domain A and 
validate on domain B
- ...

Creating your own data split is simple: the splits_final.pkl file contains the following data structure (assume there are five training cases A, B, C, D, and E):
```python
splits = [
    {'train': ['A', 'B', 'C', 'D'], 'val': ['E']},
    {'train': ['A', 'B', 'C', 'E'], 'val': ['D']},
    {'train': ['A', 'B', 'D', 'E'], 'val': ['C']},
    {'train': ['A', 'C', 'D', 'E'], 'val': ['B']},
    {'train': ['B', 'C', 'D', 'E'], 'val': ['A']}
]
```

Use load_pickle and save_pickle from batchgenerators.utilities.file_and_folder_operations for loading/storing the splits.

Splits is a list of length NUMBER_OF_FOLDS. Each entry in the list is a dict, with 'train' and 'val' as keys and lists 
of the corresponding case names (without the _0000 etc!) as values.

nnU-Net's five-fold cross validation will always create a list of len(splits)=5. But you can do whatever you want. Note 
that if you define only 4 splits (fold 0-3) and then set fold=4 when training (that would be the fifth split), 
nnU-Net will print a warning and proceed to use a random 80:20 data split. 

## How can I swap component XXX (for example the loss) of nnU-Net?

All changes in nnU-Net are handled the same way:

1) create a new nnU-Net trainer class. Place the file somewhere in the nnunet.training.network_training folder 
(any subfolder will do. If you create a new subfolder, make sure to include an empty `__init__.py` file!)

2) make your new trainer class derive from the trainer you would like to change (most likely this is going to be nnUNetTrainerV2)

3) identify the function that you need to overwrite. You may have to go up the inheritance hierarchy to find it!

4) overwrite that function in your custom trainer, make sure whatever you do is compatible with the rest of nnU-Net

What these changes need to look like specifically is hard to say without knowing what you are exactly trying to do. 
Before you open a new issue on GitHub, please have a look around the `nnunet.training.network_training` folder first! 
There are tons of examples modifying various parts of the pipeline.

Also see [here](extending_nnunet.md)

## How does nnU-Net handle multi-modal images?

Multi-modal images are treated as color channels. BraTS, which comes with T1, T1c, T2 and Flair images for each 
training case will thus for example have 4 input channels.

## Why does nnU-Net not use all my GPU memory?

nnU-net and all its parameters are optimized for a training setting that uses about 8GB of VRAM for a network training. 
Using more VRAM will not speed up the training. Using more VRAM has also not (yet) been beneficial for model 
performance consistently enough to make that the default. If you really want to train with more VRAM, you can do one of these things:

1) Manually edit the plans files to increase the batch size. A larger batch size gives better (less noisy) gradients 
and may improve your model performance if the dataset is large. Note that nnU-Net always runs for 1000 epochs with 250 
iterations each (25000 iterations). The training time thus scales approximately linearly with the batch size 
(batch size 4 is going to need twice as long for training than batch size 2!)

2) Manually edit the plans files to increase the patch size. This one is tricky and should only been attempted if you 
know what you are doing! Again, training times will be increased if you do this! 3) is a better way of increasing the 
patch size.

3) Run `nnUNet_plan_and_preprocess` with a larger GPU memory budget. This will make nnU-Net plan for larger patch sizes 
during experiment planning. Doing this can change the patch size, network topology, the batch size as well as the 
presence of the U-Net cascade. To run with a different memory budget, you need to specify a different experiment planner, for example
`nnUNet_plan_and_preprocess -t TASK_ID -pl2d None -pl3d ExperimentPlanner3D_v21_32GB` (note that `-pl2d None` will 
disable 2D U-Net configuration. There is currently no planner for larger 2D U-Nets). We have planners for 8 GB (default), 
11GB and 32GB available. If you need a planner for a different GPU size, you should be able to quickly hack together 
your own using the code of the 11GB or 32GB planner (same goes for a 2D planner). Note that we have experimented with 
these planners and not found an increase in segmentation performance as a result of using them. Training times are 
again longer than with the default.

## Do I need to always run all U-Net configurations?
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
 
## Sharing Models
You can share trained models by simply sending the corresponding output folder from `RESULTS_FOLDER/nnUNet` to 
whoever you want share them with. The recipient can then use nnU-Net for inference with this model.

You can now also use `nnUNet_export_model_to_zip` to export a trained model (or models) to a zip file. The recipient 
can then use `nnUNet_install_pretrained_model_from_zip` to install the model from this zip file. 

## Can I run nnU-Net on smaller GPUs?
nnU-Net is guaranteed to run on GPUs with 11GB of memory. Many configurations may also run on 8 GB.
If you have an 11GB and there is still an `Out of Memory` error, please read 'nnU-Net training: RuntimeError: CUDA out of memory' [here](common_problems_and_solutions.md).
 
If you wish to configure nnU-Net to use a different amount of GPU memory, simply adapt the reference value for the GPU memory estimation 
accordingly (with some slack because the whole thing is not an exact science!). For example, in 
[experiment_planner_baseline_3DUNet_v21_11GB.py](nnunet/experiment_planning/experiment_planner_baseline_3DUNet_v21_11GB.py) 
we provide an example that attempts to maximise the usage of GPU memory on 11GB as opposed to the default which leaves 
much more headroom). This is simply achieved by this line:

```python
ref = Generic_UNet.use_this_for_batch_size_computation_3D * 11 / 8
```

with 8 being what is currently used (approximately) and 11 being the target. Should you get CUDA out of memory 
issues, simply reduce the reference value. You should do this adaptation as part of a separate ExperimentPlanner class. 
Please read the instructions [here](documentation/extending_nnunet.md).


## Why is no 3d_lowres model created?
3d_lowres is created only if the patch size in 3d_fullres less than 1/8 of the voxels of the median shape of the data 
in 3d_fullres (for example Liver is about 512x512x512 and the patch size is 128x128x128, so that's 1/64 and thus 
3d_lowres is created). You can enforce the creation of 3d_lowres models for smaller datasets by changing the value of
`HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0` (located in experiment_planning.configuration).
    
