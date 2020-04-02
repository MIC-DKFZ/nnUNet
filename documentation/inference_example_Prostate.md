# Example: inference with pretrained nnU-Net models

This is a step-by-step example on how to run inference with pretrained nnU-Net models on the Prostate dataset of the 
Medical Segemtnation Decathlon.

1) Install nnU-Net by following the instructions [here](../readme.md#installation). Make sure to set all relevant paths, 
also see [here](setting_up_paths.md). This step is necessary so that nnU-Net knows where to store trained models.
2) Download the Prostate dataset of the Medical Segmentation Decathlon from 
[here](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). Then extract the archive to a 
destination of your choice.
3) We selected the Prostate dataset for this example because we have a utility script that converts the test data into 
the correct format. 

    Decathlon data come as 4D niftis. This is not compatible with nnU-Net (see dataset format specified 
    [here](dataset_conversion.md)). Convert the Prostate dataset into the correct format with

    ```bash
    nnUNet_convert_decathlon_task -i /xxx/Task05_Prostate
    ```
    
    Note that `Task05_Prostate` must be the folder that has the three 'imagesTr', 'labelsTr', 'imagesTs' subfolders!
    The converted dataset can be found in `$nnUNet_raw_data_base/nnUNet_raw_data` ($nnUNet_raw_data_base is the folder for 
    raw data that you specified during installation)
4) Download the pretrained model using this command:
    ```bash
    nnUNet_download_pretrained_model Task005_Prostate
    ```
5) The prostate dataset requires two image modalities as input. This is very much liKE RGB images have three color channels. 
nnU-Net recognizes modalities by the file ending: a single test case of the prostate dataset therefore consists of two files 
`case_0000.nii.gz` and `case_0001.nii.gz`. Each of these files is a 3D image. The file ending with 0000.nii.gz must 
always contain the T2 image and 0001.nii.gz the ADC image. Whenever you are using pretrained models, you can use
    ```bash
    nnUNet_print_pretrained_model_info Task005_Prostate
    ```
   to obtain information on which modality needs to get which number. The outpput for Prostate is the following:
    
        Prostate Segmentation. 
        Segmentation targets are peripheral and central zone, 
        input modalities are 0: T2, 1: ADC. 
        Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/
6) The script we ran in 3) automatically converted the test data for us and stored them in
`$nnUNet_raw_data_base/nnUNet_raw_data/Task005_Prostate/imagesTs`. Note that you need to to this conversion youself when 
using other than Medcial Segmentation Decathlon datasets. No worries. Doing this is easy (often as simple as appending 
a _0000 to the file name if only one input modality is required). Instructions can be found here [here](data_format_inference.md).
7) You can now predict the Prostate test cases with the pretrained model. We exemplarily use the 3D full resoltion U-Net here:
    ```bash
    nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task005_Prostate/imagesTs/ -o OUTPUT_DIRECTORY -t 5 -m 3d_fullres
    ``` 
    Note that `-t 5` specifies the task with id 5 (which corresponds to the Prostate dataset). You can also give the full 
    task name `Task005_Prostate`. `OUTPUT_DIRECTORY` is where the resulting segmentations are saved.
8) If you want to use an ensemble for inference, you need to run the following commands:

    Prediction with 3d full resolution U-Net (this command is a little different than the one above). 
    ```bash
    nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task005_Prostate/imagesTs/ -o OUTPUT_DIRECTORY_3D -t 5 --save_npz -m 3d_fullres
    ```
    
    Prediction with 2D U-Net
    ```bash
    nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task005_Prostate/imagesTs/ -o OUTPUT_DIRECTORY_2D -t 5 --save_npz -m 2d
    ```
    `--save_npz` will tell nnU-Net to also store the softmax probabilities for ensembling. 
    
    You can then merge the predictions with
    ```bash
    nnUNet_ensemble -f OUTPUT_DIRECTORY_3D OUTPUT_DIRECTORY_2D -o OUTPUT_FOLDER_ENSEMBLE -pp POSTPROCESSING_FILE
    ```
   This will merge the predictions from `OUTPUT_DIRECTORY_2D` and `OUTPUT_DIRECTORY_3D`. `-pp POSTPROCESSING_FILE` 
   (optional!) is a file that gives nnU-Net information on how to postprocess the ensemble. These files were also 
   downloaded as part of the pretrained model weights and are located at `RESULTS_FOLDER/nnUNet/ensembles/
   Task005_Prostate/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json`. 
   We will make the postprocessing files more accssible in a future (soon!) release.