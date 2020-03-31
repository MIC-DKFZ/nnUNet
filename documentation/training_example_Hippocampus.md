# Example: 3D U-Net training on the Hippocampus dataset
 
This is a step-by-step example on how to run a 3D full resolution Training with the Hippocampus dataset from the 
Medical Segmentation Decathlon.

1) Install nnU-Net by following the instructions [here](../readme.md#installation). Make sure to set all relevant paths, 
also see [here](setting_up_paths.md). This step is necessary so that nnU-Net knows where to store raw data, 
preprocessed data and trained models.
2) Download the Hippocampus dataset of the Medical Segmentation Decathlon from 
[here](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). Then extract the archive to a 
destination of your choice.
3) Decathlon data come as 4D niftis. This is not compatible with nnU-Net (see dataset format specified 
[here](dataset_conversion.md)). Convert the Hippocampus dataset into the correct format with

    ```bash
    nnUNet_convert_decathlon_task -i /xxx/Task04_Hippocampus
    ```
    
    Note that `Task04_Hippocampus` must be the folder that has the three 'imagesTr', 'labelsTr', 'imagesTs' subfolders!
    The converted dataset can be found in nnUNet_raw_data_base/nnUNet_raw_data (nnUNet_raw_data_base is the folder for 
    raw data that you specified during installation)
4) You can now run nnU-Nets pipeline configuration (and the preprocessing) with the following line:
    ```bash
    nnUNet_plan_and_preprocess -t 4
    ```
   Where 4 refers to the task ID of the Hippocampus dataset.
5) Now you can already start network training. This is how you train a 3d full resoltion U-Net on the Hippocampus dataset:
    ```bash
    nnUNet_train 3d_fullres nnUNetTrainerV2 4 0
    ```
   nnU-Net per default requires all trainings as 5-fold cross validation. The command above will run only the training for the 
   first fold (fold 0). 4 is the task identifier of the hippocampus dataset. Training one fold should take about 20 
   hours on a modern GPU.
   
This tutorial is only intended to demonstrate how easy it is to get nnU-Net running. You do not need to finish the 
network training - pretrained models for the hippocampus task are available (see [here](../readme.md#run-inference)).

The only prerequisite for running nnU-Net on your custom dataset is to bring it into a structured, nnU-Net compatible 
format. nnU-Net will take care of the rest. See [here](dataset_conversion.md) for instructions on how to convert 
datasets into nnU-Net compatible format.
