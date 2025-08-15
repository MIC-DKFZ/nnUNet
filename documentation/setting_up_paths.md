# Setting up Paths

nnU-Net relies on environment variables to know where raw data, preprocessed data and trained model weights are stored. 
To use the full functionality of nnU-Net, the following three environment variables must be set:

1) `nnUNet_raw`: This is where you place the raw datasets. This folder will have one subfolder for each dataset names 
DatasetXXX_YYY where XXX is a 3-digit identifier (such as 001, 002, 043, 999, ...) and YYY is the (unique) 
dataset name. The datasets must be in nnU-Net format, see [here](dataset_format.md).

    Example tree structure:
    ```text
    nnUNet_raw/Dataset001_NAME1
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    ├── imagesTs
    │   ├── ...
    └── labelsTr
        ├── ...
    nnUNet_raw/Dataset002_NAME2
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

See [here](set_environment_variables.md).
