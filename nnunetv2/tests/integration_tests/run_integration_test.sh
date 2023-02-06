

nnUNetv2_train $1 3d_fullres 0 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_fullres 1 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_fullres 2 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_fullres 3 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_fullres 4 -tr nnUNetTrainer_5epochs --npz

nnUNetv2_train $1 2d 0 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 2d 1 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 2d 2 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 2d 3 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 2d 4 -tr nnUNetTrainer_5epochs --npz

nnUNetv2_train $1 3d_lowres 0 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_lowres 1 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_lowres 2 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_lowres 3 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_lowres 4 -tr nnUNetTrainer_5epochs --npz

nnUNetv2_train $1 3d_cascade_fullres 0 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_cascade_fullres 1 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_cascade_fullres 2 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_cascade_fullres 3 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train $1 3d_cascade_fullres 4 -tr nnUNetTrainer_5epochs --npz

python nnunetv2/tests/integration_tests/run_integration_test_bestconfig_inference.py -d $1