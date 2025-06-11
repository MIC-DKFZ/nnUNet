#!bin/bash/

export nnUNet_raw="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_results"
CUDA_VISIBLE_DEVICES=0 uv run --extra cu124 nnUNetv2_train 1 3d_fullres 0 --npz -device 'cuda' -num_gpus 1 -p nnUNetResEncUNetMPlans