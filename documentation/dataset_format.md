# nnU-Net dataset format
The only way to bring your data into nnU-Net is by storing it in a specific format. Due to nnU-Net's roots in the
[Medical Segmentation Decathlon](http://medicaldecathlon.com/) (MSD), its dataset is heavily inspired (but has since 
diverged, see also [here](#how-to-use-decathlon-datasets)) from the format used in the MSD.

Datasets consist of three components: raw images, corresponding segmentation maps and a dataset.json file specifying 
some metadata. 

## What do training cases look like?
Each training case is associated with an identifier = a unique name for that case. This identifier is used by nnU-Net to 
connect images with the correct segmentation.

A training case consists of images and a corresponding segmentation. 

**Images** is plural because nnU-Net supports arbitrarily many input channels. In order to be as flexible as possible, 
nnU-net requires each input channel to be stored in a separate image (with the sole exception being RGB natural 
images). So these images could for example be a T1 and a T2 MRI (or whatever else you want). The different input 
channels MUST have the same geometry (same shape, spacing (if applicable) etc) and
must be co-registered (if applicable). Input channels are identified by nnU-Net by their suffix: a four-digit integer at the end 
of the filename. Image files must therefore follow the following naming convention: case_identifier_XXXX.SUFFIX. 
Hereby, XXXX is the modality identifier and SUFFIX is the file extension used by your image format (png, nii.gz, ...). 
The dataset.json file connects channel names with the channel identifiers in the 'channel_names' key. See below.

**Segmentations** must share the same geometry with their corresponding images (same shape etc). Segmentations are 
integer maps with each value representing a semantic class. The background must be 0. If there is no background, then 
do not use the label 0 for something else! Integer values of your semantic classes must be consecutive (0, 1, 2, 3, 
...). Of course, not all labels must be present in each training case. Segmentations are saved as case_identifier.SUFFIX

Within a training case, all image geometries (input channels, corresponding segmentation) must match. Between training 
cases, they can of course differ. nnU-Net takes care of that.

Important: The input channels must be consistent! Concretely, **all images need the same input channels in the same 
order**. This is also true for inference! 

## Supported file formats

One big change in nnU-Net V2 is the support of multiple input file types. Gone are the days of converting everything to .nii.gz!
This is implemented by abstracting the input and output of images + segmentations through `BaseReaderWriter`. nnU-Net 
comes with a broad collection of Readers+Writers and you can even add your own to support your data format! 
See [here](../nnunetv2/imageio/readme.md).

As a nice bonus, nnU-Net now also natively supports 2D input images and you no longer have to mess around with 
conversions to pseudo 3D niftis. Yuck. That was disgusting.

By default, the following file formats are supported:
- NaturalImage2DIO: .png, .bmp, .tif
- NibabelIO: .nii.gz, .nrrd, .mha
- NibabelIOWithReorient: .nii.gz, .nrrd, .mha. This reader will reorient images to RAS!
- SimpleITKIO: .nii.gz, .nrrd, .mha
- Tiff3DIO: .tif, .tiff. 3D tif images!

The file extension lists are not exhaustive and depend on what the backend supports. For example, nibabel and SimpleITK 
support more than the three given here. The file endings given here are just the ones we tested!

IMPORTANT: nnU-Net can only be used with file formats that use lossless (or no) compression! Because the file 
format is defined for an entire dataset (and not separately for images and segmentations, this could be a todo for 
the future), we must ensure that there are no compression artifacts that destroy the segmentation maps. So no .jpg and 
the likes! 

## Dataset folder structure
Datasets must be located in the `nnUNet_raw` folder (which you need to define when installing nnU-Net!).
Each segmentation dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit 
integer, and a dataset name (which you can freely choose): Dataset005_Prostate has 'Prostate' as dataset name and 
the dataset id is 5. Datasets are stored in the `nnUNet_raw` folder like this:

    nnUNet_raw/
    ├── Dataset001_BrainTumour
    ├── Dataset002_Heart
    ├── Dataset003_Liver
    ├── Dataset004_Hippocampus
    ├── Dataset005_Prostate
    ├── ...

Within each dataset folder, the following structure is expected:

    Dataset001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs  # optional
    └── labelsTr


When adding your custom dataset, take a look at the [dataset_conversion](../nnunetv2/dataset_conversion) folder and 
pick an id that is not already taken. IDs 1-10 are Medical Segmentation decathlon.

- **imagesTr** contains the images belonging to the training cases. nnU-Net will run pipeline configuration, training with 
cross-validation, as well as finding postprocesing and the best ensemble on this data. 
- **imagesTs** (optional) contains the images that belong to the test cases. nnU-Net does not use them! This could just 
be a convenient location for you to store these images.
- **labelsTr** contains the images with the ground truth segmentation maps for the training cases. 
- **dataset.json** contains metadata of the dataset.

The scheme introduced [above](#what-do-training-cases-look-like) results in the following folder structure. Given 
is an example for the first Dataset of the MSD: BrainTumour. This dataset hat four input channels: FLAIR (0000), 
T1w (0001), T1gd (0002) and T2w (0003). Note that the imagesTs folder is optional and does not have to be present.

    nnUNet_raw_data_base/nnUNet_raw_data/Dataset001_BrainTumour/
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
    │   ├── ...
    └── labelsTr
        ├── BRATS_001.nii.gz
        ├── BRATS_002.nii.gz
        ├── ...

Here is another example of the second dataset of the MSD, which has only one input channel:

    nnUNet_raw_data_base/nnUNet_raw_data/Dataset002_Heart/
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

Remember: For each training case, all images must have the same geometry to ensure that their pixel arrays are aligned. Also 
make sure that all your data is co-registered!

See also [dataset format inference](dataset_format_inference.md)!!

## dataset.json
The dataset.json contains metadata that nnU-Net needs for training. We have greatly reduced the number of required 
fields since version 1!

Here is what the dataset.json should look like at the example of the Dataset005_Prostate from the MSD:

    { 
     "channel_names": {  # formerly modalities
       "0": "T2", 
       "1": "ADC"
     }, 
     "labels": {  # THIS IS DIFFERENT NOW!
       "background": 0,
       "PZ": 1,
       "TZ": 2
     }, 
     "numTraining": 32, 
     "file_ending": ".nii.gz"
     "overwrite_image_reader_writer": "SimpleITKIO"  # optional! If not provided nnU-Net will automatically determine the ReaderWriter
     }

The channel_names determine the normalization used by nnU-Net. If a channel is marked as 'CT', then a global 
normalization based on the intensities in the foreground pixels will be used. If it is something else, per-channel 
z-scoring will be used. See our paper for more details. nnU-Net v2 intoduces a few more normalization schemes to 
choose from and allows you to define your own, see [here](explanation_normalization.md) for more information. 

Important changes relative to nnU-Net V1:
- "modality" is now called "channel_names" to remove strong bias to medical images
- labels are structured differently (name -> int instead if int -> name). This was needed to support [region-based training](region_based_training.md)
- "file_ending" is added to support different input file types
- "overwrite_image_reader_writer" optional! Can be used to specify a certain (custom) ReaderWriter class that should 
be used with this dataset. If not provided, nnU-Net will automatically determine the ReaderWriter
- "regions_class_order" only used in [region-based training](region_based_training.md)

There is a utility with which you can generate the dataset.json automatically. You can find it 
[here](../nnunetv2/dataset_conversion/generate_dataset_json.py). 
See our examples in [dataset_conversion](../nnunetv2/dataset_conversion) for how to use it. And read its documentation!

## How to use decathlon datasets
See [convert_msd_dataset.md](convert_msd_dataset.md)

## How to use 2D data with nnU-Net
2D is now natively supported *yayy*. See [here](#supported-file-formats) as well as the example dataset in this 
[script](../nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py).


## How to update an existing dataset
When updating a dataset it is best practice to remove the preprocessed data in `nnUNet_preprocessed/DatasetXXX_NAME` 
to ensure a fresh start. Then replace the data in `nnUNet_raw` and rerun `nnUNetv2_plan_and_preprocess`. Optionally 
also remove the results from any old trainings.

# Example dataset conversion scripts
In the `dataset_conversion` folder (see [here](../nnunetv2/dataset_conversion)) are multiple example scripts for 
converting datasets into nnU-Net format. These scripts cannot be run as they are (you need to open them and change 
some paths) but they are excellent examples for you to learn how to convert your own datasets into nnU-Net format. 
Just pick the dataset that is closest to yours as a starting point.
The list of dataset conversion scripts is continually updated. If you find that some publicly available dataset is 
missing, feel free to open a PR to add it!
