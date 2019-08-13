#!/bin/bash

python3 rename_input.py

echo '==============================='
echo 'Run nnUnet Prediction'
echo '==============================='
cd nnUNet/nnunet
# Change the call according to your parameters.
python3 inference/predict_simple.py -i $INPUTDIR -o $OUTPUTDIR -t $TASK_NAME -tr nnUNetTrainer -m 3d_fullres
