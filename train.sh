#!/bin/bash
source ~/env_file

eval "$(conda shell.bash hook)"
conda activate generic-nnunet

cd /home/stud/strasser/workspace/nnUNet && git checkout dev

echo "starting training with fold $fold"
nnUNetv2_train 42 3d_fullres  $fold --npz
