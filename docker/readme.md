## nnUNet Docker Container (Inference)

This container takes the [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) code and inherits from a cuda-10.0-runtime Docker image. Thus, the execution needs the docker nvidia tools to be runnable.

### Environment-Parameters
> - `INPUTDIR`: dir path to the input folder containing the compressed nifti files (nii.gz):
> ```
> /inputdir/
>  - imageA.nii.gz
>  - imageB.nii.gz
> ```
> The nifti files are renamed by the container, so you don't have to rename them before.
> - `OUTPUTDIR`: dir path to the output folder containing all segmentations which were calculated during the container process

> - `RESULTS_FOLDER`: this is very important to make right, because here you have to mount a directory containing your model dir tree starting from the 'nnUNet'-folder
> Example: `\opt\path\data\nnUNet` is the correct path when your model lives here: `\opt\path\data\nnUNet\3d_fullres\Task17_AbdominalOrganSegmentation\nnUNetTrainer__nnUNetPlans`,

> - `TASK_NAME`: this name should match the task name you used for training the model. The task name appears also within the RESULTS_FOLDER variable

___
You may want to change data loading or output structuring. This shows an example use which can be adjusted to a specific need.
