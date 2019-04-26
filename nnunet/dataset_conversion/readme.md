# Dataset conversion instructions

## How to use decathlon datasets

First make sure you have all the proper paths set in `paths.py`: base, preprocessing_output_dir and network_training_output_dir.
Then copy the downloaded dataset into `base/nnUNet_raw`. For Task04_Hippocampus, for example, this should look like this:
`base/nnUNet_raw/Task04_Hippocampus`. Hereby, Task04_Hippocampus has three subfolders 
(`imagesTr`, `labelsTr`, `imagesTs`) and a `dataset.json` file.

You can run the preprocessing and experiment planning for this stask by executing

`python experiment_planning/plan_and_preprocess_task.py -t Task04_Hippocampus -p 8`


For historical reasons nnU-Net does not like 4D niftis, so the first preprocessing step done by nnU-Net will be splitting the 
4D niftis into a series of 3D niftis. They will be stored in `base/nnUNet_raw_splitted` when you run the preprocessing.
You can skip the FSL dependency by doing the splitting manually. nnU-Net will detect that splitted data is available and
not attempt to re-run it. Instructions of how to do this are provided below.

## How to convert non-Decathlon datasets for nnU-Net
In this file we provide a description on how you need to convert your dataset to make it compatible with nnU-Net.

For the purpose of this 
manual, we refer to your dataset as `TaskXX_MY_DATASET`. Hereby, `XX` is is a dual-digit number and `MY_DATASET` can
be anything you want.

We follow the folder structure of the Medical Segmentation Decathlon (MSD). In the `splitted_4d_output_dir` (see `paths.py`)
, create a subfolder
called `TaskXX_MY_DATASET`. In that subfolder, create the following three directories: `imagesTr`, `labelsTr`(, `imagesTs`). 
These are for training images, training labels (and test images), respectively.  

Just like the MSD we use the nifti (.nii.gz) file format. There is a crucial difference though.
While the MSD provides 4D niftis where the first axis is for the modality, we prefer to split each training case into 
separate 3D Niftis. So what was a File `patientID.nii.gz` containing an image of shape (4, 160, 190, 160) now is four files with shape 
(160, 190, 160) each. These four files should be named `patientID_0000.nii.gz`, `patientID_0001.nii.gz`, `patientID_0002.nii.gz`, 
`patientID_0003.nii.gz`, where the ending 4-digit represents the modality. `patientID` can be anything you want. Make 
sure that the same modality is assigned the same digit for all patients (for example if you have T1 and T2 then you 
need to make sure T1 is always 0000 and T2 always 0001). If you data has only one modality, just name your cases `patientID_0000.nii.gz`.

Copy all training cases (just the data, not the labels) into the `imagesTr` subfolder.

Copy training labels into the `labelsTr` subfolder. Labels are always 3D niftis and should be named `patientID.nii.gz` 
where `patientiID` is identical to the identifier used for the corresponding raw data.

If you have test images, convert them to 3D Nifti like the training data and place them in the `imagesTs` folder. 
All data you wish to predict with nnU-Net must be in this format.

Finally you need to create a `dataset.json` file that you place in the `TaskXX_MY_DATASET` root folder. This file was 
always provided in the MSD challenge and is therefore a requirement for nnU-Net. Have a look at (for example) 
`LiverTumorSegmentationChallenge.py` to see what it needs to look like. Important: The list stored in 'training' contains images and labels are 
both called `patientID.nii.gz` (dropping the 0000 ending we used here) for consistency!