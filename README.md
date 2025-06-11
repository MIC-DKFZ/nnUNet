# M31 Technical Assessment

## Content
- `playground.ipynb`: Prototype code for model architecture.

## Run nnUnetV2
1. Specify environment variables for data paths
```bash
export nnUNet_raw="<your-project-root-directory>/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="<your-project-root-directory>/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="<your-project-root-directory>/nnunet_data/nnUNet_results"
```

Mine version:
```bash
export nnUNet_raw="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_results"
```
2. Setup paths
```bash
# example path structure
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
3. Train nnUnetV2
**All `nnUnetV2` commands are in the env path, so you can execute them anywhere with the venv activated, or ran with the `uv run --extra [cu121, cu124, cu128] <your command>` command. The commands uses `nnUnetv2_` prefix.**
```bash
# train command (recommend to run this in the terminal)
uv run --extra cu<cuda-version> nnUNetv2_train <dataset_id> <3d_lowres | 3d_fullres> <crossvalidataion_fold_index> --npz -device 'cuda' --c <checkpoint_path>

# mine example
CUDA_VISIBLE_DEVICES=0 uv run --extra cu124 nnUNetv2_train 1 3d_fullres 0 --npz -device 'cuda' -num_gpus 1
```
Some other flags that might be useful:
```bash
-tr TR                [OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer
-p P                  [OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans
```

## Run training and prediction on ResEncPlans
```
uv run --extra cu<cuda-version> nnUNetv2_train <dataset_id> <3d_lowres | 3d_fullres> <crossvalidataion_fold_index> --npz -device 'cuda' --c <checkpoint_path> -p nnUNetResEncUNet(M/L/XL)Plans

# mine example
CUDA_VISIBLE_DEVICES=0 uv run --extra cu124 nnUNetv2_train 1 3d_fullres 0 --npz -device 'cuda' -num_gpus 1 -p nnUNetResEncUNetMPlans
```