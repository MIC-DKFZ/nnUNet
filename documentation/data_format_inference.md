# Data format for Inference 

The data format for inference must match the one used for the raw data (specifically, the images must be in exactly 
the same format as in the imagesTr folder). As before, the filenames must start with a
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

    _0000 is always the T2 image and _0001 is always the ADC image (as specified by 'modality' in the dataset.json)

2) Task002_Heart:

        imagesTs
        ├── la_001_0000.nii.gz
        ├── la_002_0000.nii.gz
        ├── la_006_0000.nii.gz
        ├── ...
    
    Task002 only has one modality, so each case only has one _0000.nii.gz file.
  

The segmentations in the output folder will be named INDENTIFIER.nii.gz (omitting the modality identifier).
   