# Setting up paths.py
nnU-Net relies on environment variables to know where raw data, preprocessed data and trained model weights are stored. 
To use the full functionality of nnU-Net, the following three environment variables must be set:

1) nnUNet_raw_data_base: This is there nnU-Net finds the raw data and stored the cropped data. The folder located at 
nnUNet_raw_data_base must have at least the subfolder nnUNet_raw_data, which in turn contains one subfolder for each Task. 
It is the responsibility of the user to bring the raw data into the appropriate format - nnU-Net will then take care of 
the rest ;-) For more information on the required raw data format, see [here](dataset_conversion/readme.md).

    Example tree structure:
    ```
    nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart
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
    nnUNet_raw_data_base/nnUNet_raw_data/Task005_Prostate/
    ├── dataset.json
    ├── imagesTr
    │   ├── prostate_00_0000.nii.gz
    │   ├── prostate_00_0001.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── prostate_03_0000.nii.gz
    │   ├── prostate_03_0001.nii.gz
    │   ├── ...
    └── labelsTr
        ├── prostate_00.nii.gz
        ├── prostate_01.nii.gz
        ├── ...
    ```

2) nnUNet_preprocessed: This is the folder where the preprocessed data will be saved. The data will also be read from 
this folder during training. Therefore it is important that it is located on a drive with low access latency and high 
throughput (a regular sata or nvme SSD is sufficient).

3) RESULTS_FOLDER: This specifies where nnU-Net will save the model weights. If pretrained models are downloaded, this 
is where it will save them.

### How to set environment variables
(nnU-Net was developed for Ubuntu/Linux. The following guide is intended for this operating system and will not work on 
others. We do not provide support for other operating systems!)

There are several ways you can do this. The most common one is to set the paths in your .bashrc file, which is located 
in your home directory. For me, this file is located at /home/fabian/.bashrc. You can open it with any text editor of 
choice. If you do not see the file, that may be because it is hidden by default. You can run `ls -al /home/fabian` to 
ensure that you see it. In rare cases it may not be present and you can simply create it with `touch /home/fabian/.bashrc`.

Once the file is open in a text editor, add the following lines to the bottom:
```
export nnUNet_raw_data_base="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export RESULTS_FOLDER="/media/fabian/nnUNet_trained_models"
```

(of course adapt the paths to your system and remember that nnUNet_preprocessed should be located on an SSD!)

Then save and exit. To be save, make sure to reload the .bashrc by running `source /home/fabian/.bashrc`. Reloading 
needs only be done on terminal sessions that were already open before you saved the changes. Any new terminal you open 
after will have these paths set. You can verify that the paths are set up properly by typing `echo $RESULTS_FOLDER` 
etc and it should print out the correct folder.

### An alternative way of setting these paths
The method above sets the paths permanently (until you delete the lines from your .bashrc) on your system. If you wish 
to set them only temporarily, you can run the export commands in your terminal:

```
export nnUNet_raw_data_base="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export RESULTS_FOLDER="/media/fabian/nnUNet_trained_models"
```

This will set the paths for the current terminal session only (the variables will be lost if you close the terminal 
and need to be reset every time).