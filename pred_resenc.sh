#!bin/bash/

export nnUNet_raw="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_results"
CUDA_VISIBLE_DEVICES=0 uv run --extra cu124 nnUNetv2_predict \
  -i /mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_raw/Dataset001_PancreasSegClassification/imagesTs \
  -o /mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_results/Dataset001_PancreasSegClassification/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/val_predictions \
  -d 001 \
  -c 3d_fullres \
  -f 0 \
  -p nnUNetResEncUNetMPlans \
  -chk checkpoint_best.pth \
  --verbose