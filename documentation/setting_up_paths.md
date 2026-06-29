# Setting up Paths

Prefer the consolidated setup guide for the recommended path:

- [Installation and setup](getting-started/installation-and-setup.md)

nnU-Net relies on environment variables to know where raw data, preprocessed data and trained model weights are stored.
To use the full functionality of nnU-Net, the following three environment variables must be set:

1) `nnUNet_raw`: This is where you place the raw datasets. This folder will have one subfolder for each dataset names
DatasetXXX_YYY where XXX is a 3-digit identifier (such as 001, 002, 043, 999, ...) and YYY is the (unique)
dataset name. The datasets must be in nnU-Net format, see [here](dataset_format.md).

    Example tree structure:
    ```
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

### Optional: `nnUNet_extTrainer`

If you need to load a custom `nnUNetTrainer` subclass from outside the `nnunetv2` package
(for example to run inference with a checkpoint someone else trained), set
`nnUNet_extTrainer` to one or more directories that contain the trainer code, separated
by the OS path separator (`:` on Linux/macOS, `;` on Windows).

See [Share models trained with a custom trainer](how-to/share-models-with-custom-trainers.md)
for the full guide.

### How to set environment variables

See [here](set_environment_variables.md).
