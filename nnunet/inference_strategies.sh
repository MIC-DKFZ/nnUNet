python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_normal -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -p nnUNetPlans_customClip -f 0 -tr nnUNetTrainer --tta 0 --num_threads_preprocessing 2 --num_threads_nifti_save 6 --mode normal --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_normal_2folds -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -p nnUNetPlans_customClip -f 0 1 -tr nnUNetTrainer --tta 0 --num_threads_preprocessing 2 --num_threads_nifti_save 6 --mode normal --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_normal_all_in_gpu_False -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -p nnUNetPlans_customClip -f 0 -tr nnUNetTrainer --tta 0 --num_threads_preprocessing 2 --num_threads_nifti_save 6 --mode normal --all_in_gpu False

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_fast -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -p nnUNetPlans_customClip -f 0 -tr nnUNetTrainer --tta 0 --num_threads_preprocessing 2 --num_threads_nifti_save 6 --mode fast --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_fastest -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -p nnUNetPlans_customClip -f 0 -tr nnUNetTrainer --tta 0 --num_threads_preprocessing 2 --num_threads_nifti_save 6 --mode fastest --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_fastest_2folds -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -p nnUNetPlans_customClip -f 0 1 -tr nnUNetTrainer --tta 0 --num_threads_preprocessing 2 --num_threads_nifti_save 6 --mode fastest --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_fast -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror -p nnUNetPlans_customClip --tta 0 --num_threads_preprocessing 3 --num_threads_nifti_save 6 --mode fast --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_fast_5folds -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainerV2_2_noMirror -p nnUNetPlans_customClip --tta 0 --num_threads_preprocessing 3 --num_threads_nifti_save 6 --mode fast --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_fast_lowres -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_lowres -f 0 -tr nnUNetTrainerV2_2_noMirror -p nnUNetPlans_customClip --tta 0 --num_threads_preprocessing 3 --num_threads_nifti_save 6 --mode fast --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_lowres_normal -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_lowres -f 0 -tr nnUNetTrainerV2_2_noMirror -p nnUNetPlans_customClip --tta 0 --num_threads_preprocessing 3 --num_threads_nifti_save 6 --mode normal --all_in_gpu True

python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_fast_lowres_5folds -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_lowres -f 0 1 2 3 4 -tr nnUNetTrainerV2_2_noMirror -p nnUNetPlans_customClip --tta 0 --num_threads_preprocessing 3 --num_threads_nifti_save 6 --mode fast --all_in_gpu True


