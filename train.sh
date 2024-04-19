#!/bin/bash
source ~/env_file

eval "$(conda shell.bash hook)"
conda activate generic-nnunet


echo "starting training with fold $fold"
nnUNetv2_train 137 3d_fullres  $fold --npz --c
