# Dataset conversion instructions
nnU-Net requires the raw data to be brought into a specific format so that it know how to read and interpret it. This 
format closely, but not entirely, follows the format used by the 
[Medical Segmentation Decathlon](http://medicaldecathlon.com/) (MSD).

The entry point to nnU-Net is the nnUNet_raw_data_base folder (which the user needs to specify when installing nnU-Net!). 
Each segmentation dataset is stored as a separate 'Task'. Tasks are associated with a task ID, a three digit integer 
(this is different from the MSD!) and 
a task name (which you can freely chosse): Task0005_Prostate has 'Prostate' as task name and the task id is 5. Tasks are stored in the 
nnUNet_raw_data_base/nnUNet_raw_data folder like this:

    nnUNet_raw_data_base/nnUNet_raw_data/
    ├── Task001_BrainTumour
    ├── Task002_Heart
    ├── Task003_Liver
    ├── Task004_Hippocampus
    ├── Task005_Prostate
    ├── ...

Within each task folder, the following structure is expected:

    Task001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    ├── (imagesTs)
    └── labelsTr
    
**Please make your custom task ids start at 100 to ensure that there will be no conflicts with downloaded pretrained models!!!**

imagesTr contains the images belonging to the training cases. nnU-Net will run pipeline configuration, training with 
cross-validation, as well as finding postprocesing and the best ensemble on this data. imagesTs (optional) contains the 
images that belong to the 
test cases , labelsTr the images with the ground truth segmentation maps for the training cases. dataset.json contains 
metadata of the dataset.

Each training case is associated with an identifier = a unique name for that case. This identifier is used by nnU-Net to 
recognize which label file belongs to which image. **All images (including labels) must be 3D nifti files (.nii.gz)!**
 
The image files can have any scalar pixel type. The label files must contain segmentation maps that contain consecutive integers, 
starting with 0: 0, 1, 2, 3, ... num_labels. 0 is considered background. Each class then has its own associated integer 
value.
Images may have multiple modalities. This is especially often the case for medical images. Modalities are very much 
like color channels in photos (three color channels: red, green blue), but can be much more diverse: CT, different types 
or MRI, and many other. Imaging modalities are identified by nnU-Net by their suffix: a four-digit integer at the end 
of the filename. Imaging files must therefore follow the following naming convention: case_identifier_XXXX.nii.gz. 
Hereby, XXXX is the modality identifier. What modalities these identifiers belong to is specified in the dataset.json 
file (see below). Label files are saved as case_identifier.nii.gz

This naming scheme results in the following folder structure. It is the responsibility of the user to bring their 
data into this format!

Here is an example for the first Task of the MSD: BrainTumour. Each image has four modalities: FLAIR (0000), 
T1w (0001), T1gd (0002) and T2w (0003). Note that the imagesTs folder is optional and does not have to be present.

    nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    │   ├── BRATS_001_0000.nii.gz
    │   ├── BRATS_001_0001.nii.gz
    │   ├── BRATS_001_0002.nii.gz
    │   ├── BRATS_001_0003.nii.gz
    │   ├── BRATS_002_0000.nii.gz
    │   ├── BRATS_002_0001.nii.gz
    │   ├── BRATS_002_0002.nii.gz
    │   ├── BRATS_002_0003.nii.gz
    │   ├── BRATS_003_0000.nii.gz
    │   ├── BRATS_003_0001.nii.gz
    │   ├── BRATS_003_0002.nii.gz
    │   ├── BRATS_003_0003.nii.gz
    │   ├── BRATS_004_0000.nii.gz
    │   ├── BRATS_004_0001.nii.gz
    │   ├── BRATS_004_0002.nii.gz
    │   ├── BRATS_004_0003.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── BRATS_485_0000.nii.gz
    │   ├── BRATS_485_0001.nii.gz
    │   ├── BRATS_485_0002.nii.gz
    │   ├── BRATS_485_0003.nii.gz
    │   ├── BRATS_486_0000.nii.gz
    │   ├── BRATS_486_0001.nii.gz
    │   ├── BRATS_486_0002.nii.gz
    │   ├── BRATS_486_0003.nii.gz
    │   ├── BRATS_487_0000.nii.gz
    │   ├── BRATS_487_0001.nii.gz
    │   ├── BRATS_487_0002.nii.gz
    │   ├── BRATS_487_0003.nii.gz
    │   ├── BRATS_488_0000.nii.gz
    │   ├── BRATS_488_0001.nii.gz
    │   ├── BRATS_488_0002.nii.gz
    │   ├── BRATS_488_0003.nii.gz
    │   ├── BRATS_489_0000.nii.gz
    │   ├── BRATS_489_0001.nii.gz
    │   ├── BRATS_489_0002.nii.gz
    │   ├── BRATS_489_0003.nii.gz
    │   ├── ...
    └── labelsTr
        ├── BRATS_001.nii.gz
        ├── BRATS_002.nii.gz
        ├── BRATS_003.nii.gz
        ├── BRATS_004.nii.gz
        ├── ...

Here is another example of the second task of the MSD, which has only one modality:

    nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/
    ├── dataset.json
    ├── imagesTr
    │   ├── la_003_0000.nii.gz
    │   ├── la_004_0000.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── la_001_0000.nii.gz
    │   ├── la_002_0000.nii.gz
    │   ├── ...
    └── labelsTr
        ├── la_003.nii.gz
        ├── la_004.nii.gz
        ├── ...

For each training case, all images must have the same geometry to ensure that their pixel arrays are aligned. Also 
make sure that all your data is co-registered!

The dataset.json file used by nnU-Net is identical to the ones used by the MSD. For your custom tasks you need to create 
them as well and thereby exactly follow the same structure. [This](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
is where you can download the MSD data for reference. 

Here is the content of the dataset.json from the Prostate task:

    { 
     "name": "PROSTATE", 
     "description": "Prostate transitional zone and peripheral zone segmentation",
     "reference": "Radboud University, Nijmegen Medical Centre",
     "licence":"CC-BY-SA 4.0",
     "relase":"1.0 04/05/2018",
     "tensorImageSize": "4D",
     "modality": { 
       "0": "T2", 
       "1": "ADC"
     }, 
     "labels": { 
       "0": "background", 
       "1": "PZ", 
       "2": "TZ"
     }, 
     "numTraining": 32, 
     "numTest": 16,
     "training":[{"image":"./imagesTr/prostate_16.nii.gz","label":"./labelsTr/prostate_16.nii.gz"},{"image":"./imagesTr/prostate_04.nii.gz","label":"./labelsTr/prostate_04.nii.gz"},...], 
     "test": ["./imagesTs/prostate_08.nii.gz","./imagesTs/prostate_22.nii.gz","./imagesTs/prostate_30.nii.gz",...]
     }

Note that we truncated the "training" and "test" lists for clarity. You need to specify all the cases in there. If you 
don't have test images (imagesTs does not exist) you can leave "test" blank: `"test": []`.

Please also have a look at the python files located [here](../nnunet/dataset_conversion). They show how we created our 
custom dataset.jsons for a range of public datasets.

## How to use decathlon datasets
The previous release of nnU-Net allowed userd to either start with 4D or 3D niftis. This resulted in some confusion, 
however, because some users would not know where they should save their data. We therefore dropped support for the 4D 
niftis used by the MSD. Instead, we provide a utility that converts the MSD datasets into the format specified above:

```bash
nnUNet_convert_decathlon_task -i FOLDER_TO_TASK_AS_DOWNLOADED_FROM_MSD -p NUM_PROCESSES
```

FOLDER_TO_TASK_AS_DOWNLOADED_FROM_MSD needs to point to the downloaded task folder (such as Task05_Prostate, note the 
2-digit task id!). The converted Task will be saved under the same name in nnUNet_raw_data_base/nnUNet_raw_data 
(but with a 3 digit identifier). You can overwrite the task id of the converted task by using the `-output_task_id` option.
