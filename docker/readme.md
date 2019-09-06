## nnUNet Docker Container (Inference)

This container takes the [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) code and inherits from a cuda-10.0-runtime Docker image. Thus, the execution needs the docker nvidia tools to be runnable.

This container is just used for inference and not for training.

### Input files naming
Please rename your input files beforehand by encoding the modality into the file name. This encoding must correspond with the one given by the model configuration file (JSON). Here is an example with two modalities:

```
    "modality": {
        "0": "CT",
        "1": "PET"
    },
```

This example shows a dataset containing CT scan and a PET scan for each individual with correct naming.

>  - imageA_0000.nii.gz | CT
>  - imageA_0001.nii.gz | PET
>  - imageB_0000.nii.gz | CT
>  - imageB_0001.nii.gz | PET

### Environment-Parameters
> - `INPUTDIR`: dir path to the input folder containing the compressed nifti files (nii.gz):
> - `OUTPUTDIR`: dir path to the output folder containing all segmentations which were calculated during the container process
> - `RESULTS_FOLDER`: this is very important to make right, because here you have to mount a directory containing your model dir tree starting from the 'nnUNet'-folder
> Example: `\opt\path\data\nnUNet` is the correct path when your model lives here: `\opt\path\data\nnUNet\3d_fullres\Task17_AbdominalOrganSegmentation\nnUNetTrainer__nnUNetPlans`,

> - `TASK_NAME`: this name should match the task name you used for training the model. The task name appears also within the `RESULTS_FOLDER` variable

___
You may want to change data loading or output structuring. This shows an example use which can be adjusted to a specific need.
