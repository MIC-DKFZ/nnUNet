# Data format for Inference 
Read the documentation on the overall [data format](dataset_format.md) first!

The data format for inference must match the one used for the raw data (**specifically, the images must be in exactly 
the same format as in the imagesTr folder**). As before, the filenames must start with a
unique identifier, followed by a 4-digit modality identifier. Here is an example for two different datasets:

1) Task005_Prostate:

    This task has 2 modalities, so the files in the input folder must look like this:

        input_folder
        ├── prostate_03_0000.nii.gz
        ├── prostate_03_0001.nii.gz
        ├── prostate_05_0000.nii.gz
        ├── prostate_05_0001.nii.gz
        ├── prostate_08_0000.nii.gz
        ├── prostate_08_0001.nii.gz
        ├── ...

    _0000 has to be the T2 image and _0001 has to be the ADC image (as specified by 'channel_names' in the 
dataset.json), exactly the same as what was used for training.

2) Task002_Heart:

        imagesTs
        ├── la_001_0000.nii.gz
        ├── la_002_0000.nii.gz
        ├── la_006_0000.nii.gz
        ├── ...
    
    Task002 only has one modality, so each case only has one _0000.nii.gz file.
  

The segmentations in the output folder will be named {CASE_IDENTIFIER}.nii.gz (omitting the modality identifier).

Remember that the file format used for inference (.nii.gz in this example) must be the same as was used for training 
(and as was specified in 'file_ending' in the dataset.json)!
