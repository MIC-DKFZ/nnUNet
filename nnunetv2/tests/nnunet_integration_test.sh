nnUNetv2_plan_and_preprocess -d 4

nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train 4 3d_fullres 1 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train 4 3d_fullres 2 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train 4 3d_fullres 3 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train 4 3d_fullres 4 -tr nnUNetTrainer_5epochs --npz

nnUNetv2_train 4 2d 0 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train 4 2d 1 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train 4 2d 2 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train 4 2d 3 -tr nnUNetTrainer_5epochs --npz
nnUNetv2_train 4 2d 4 -tr nnUNetTrainer_5epochs --npz

nnUNetv2_find_best_configuration 4 -c 2d 3d_fullres -tr nnUNetTrainer_5epochs -f 0 1 2 3 4

nnUNetv2_determine_postprocessing -i ${nnUNet_results}/Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4 -ref ${nnUNet_raw}/Dataset004_Hippocampus/labelsTr -np 8

nnUNetv2_apply_postprocessing -i ${nnUNet_results}/Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4 -o ${nnUNet_results}/Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4_deleteme -pp_pkl_file ${nnUNet_results}/Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8

