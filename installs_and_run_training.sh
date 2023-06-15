#!/bin/bash

# Install the required library
pip3 install albumentations==1.2.1
pip3 install -e /data/pathology/projects/pathology-lung-TIL/nnUNet_v2/

# Run the Python script
echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
echo INSTALLS DONE, START PREPROCESSING
echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
python3 /data/pathology/projects/pathology-lung-TIL/nnUNet_v2/nnunetv2/experiment_planning/experiment_planners/pathology_experiment_planner.py


echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
echo PREPROCESSING DONE, START TRAINING
echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
python3 /data/pathology/projects/pathology-lung-TIL/nnUNet_v2/nnunetv2/run/run_training.py

echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
echo TRAINING DONE, STOP
echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
echo TOTALLY DONE