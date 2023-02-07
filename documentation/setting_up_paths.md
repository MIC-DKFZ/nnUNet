# Setting up Paths

nnU-Net relies on environment variables to know where raw data, preprocessed data and trained model weights are stored. 
To use the full functionality of nnU-Net, the following three environment variables must be set:

1) `nnUNet_raw`: This is where you place the raw datasets. This folder will have one subfolder for each dataset names 
DatasetXXX_YYY where XXX is a 3-digit identifier (such as 001, 002, 043, 999, ...) and YYY is the (unique) 
dataset name. The datasets must be in nnU-Net format, see [here](dataset_format.md).

    Example tree structure:
    ```
    nnUNet_raw/nnUNet_raw_data/Dataset001_NAME1
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    ├── imagesTs
    │   ├── ...
    └── labelsTr
        ├── ...
    nnUNet_raw/nnUNet_raw_data/Dataset002_NAME2
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    ├── imagesTs
    │   ├── ...
    └── labelsTr
        ├── ...
    ```

2) `nnUNet_preprocessed`: This is the folder where the preprocessed data will be saved. The data will also be read from 
this folder during training. It is important that this folder is located on a drive with low access latency and high 
throughput (such as a nvme SSD (PCIe gen 3 is sufficient)).

3) `nnUNet_results`: This specifies where nnU-Net will save the model weights. If pretrained models are downloaded, this 
is where it will save them.

### How to set environment variables
(nnU-Net was developed for Linux. The following guide is intended for this operating system and will not work on 
others. We do not provide support for other operating systems!)

There are several ways you can do this. The most common one is to set the paths in your .bashrc file, which is located 
in your home directory. For me, this file is located at /home/isensee/.bashrc. You can open it with your text editor of 
choice. If you do not see the file, that may be because it is hidden by default. You can run `ls -al /home/isensee` to 
ensure that you see it. In rare cases it may not be present and you can simply create it with `touch /home/isensee/.bashrc`.

Once the file is open in a text editor, add the following lines to the bottom (adapt paths to your system!):
```
export nnUNet_raw="/media/fabian/nnUNet_raw_data_base"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_trained_models"
```

(of course adapt the paths to your system and remember that nnUNet_preprocessed should be located on an SSD!)

Then save and exit. Reload the .bashrc by running `source /home/fabian/.bashrc`. Reloading 
needs only be done on terminal sessions that were already open before you saved the changes. Any new terminal you open 
after will have these paths set. You can verify that the paths are set up properly by typing `echo $nnUNet_results` 
etc and it should print out the correct folder.

### An alternative way of setting these paths
The method above sets the paths permanently (until you delete the lines from your .bashrc) on your system. If you wish 
to set them only temporarily, you can run the export commands in your terminal prior to executing nnU-Net commands:

```
export nnUNet_raw="/media/fabian/nnUNet_raw_data_base"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_trained_models"
```

This will set the paths for the current terminal session only (the variables will be lost when you close the terminal 
and need to be reset every time).
