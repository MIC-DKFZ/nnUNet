# Task49 speed tests

# 3d_fullres 1 fold; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 20:30
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step2_fold0 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2

# 3d_fullres 2 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 27
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step2_fold01 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 1 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2

# 3d_fullres 3 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 34:10
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step2_fold012 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 1 2 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2

# 3d_fullres 3 folds; fast; step 1.5; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 28:09
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step15_fold012 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 1 2 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 1.5

# 3d_fullres 3 folds; fast; step 1.33333; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 22:18
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step13_fold012 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 1 2 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 1.33333333

# 3d_fullres 5 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 46:30
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step2_fold01234 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2


# Task50 speed tests
# models are still training, so the results are expected to be bad!

# 3d_fullres 1 fold; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 6
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step2_fold0 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2

# 3d_fullres 2 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 11:30
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step2_fold01 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_fullres -f 0 1 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2

# 3d_fullres 3 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 17
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step2_fold012 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_fullres -f 0 1 2 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2

# 3d_fullres 5 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip ## 27:20
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_noMirror__nnUNetPlans_customClip_fast_step2_fold01234 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2

# 3d_lowres 5 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2 ## 6:40
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_3d_lowres_nnUNetTrainerV2_2_fast_step2_fold01234 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_lowres -f 0 1 2 3 4 -tr nnUNetTrainerV2_2 --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True --step 2


# Task51 speed tests
# models are still training, so the results are expected to be bad!

# 3d_fullres 1 fold; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2 ## 5
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_fast_step2_fold0 -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2 --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True --step 2

# 3d_fullres 2 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2 ## 11
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_fast_step2_fold01 -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_fullres -f 0 1 -tr nnUNetTrainerV2_2 --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True --step 2
16
# 3d_fullres 3 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2 ## 16
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_fast_step2_fold012 -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_fullres -f 0 1 2 -tr nnUNetTrainerV2_2 --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True --step 2

# 3d_fullres 5 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2 ## 25
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_fast_step2_fold01234 -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainerV2_2 --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True --step 2

# 3d_fullres 5 folds; fast; step 1.33333333; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2 ## 14
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task51_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task51_TestSet/predictions_3d_fullres_nnUNetTrainerV2_2_fast_step13_fold01234 -t Task51_StructSeg2019_Task3_Thoracic_OAR -m 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainerV2_2 --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True --step 1.33333333333333


# Task52 speed tests
# 3d_lowres 5 folds; fast; step 2; 3pp; 6 export; all_in_gpu | model=nnUNetTrainerV2_2_noMirror ## 7
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task52_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task52_TestSet/predictions_3d_lowres_nnUNetTrainerV2_2_noMirror_fast_step2_fold01234 -t Task52_StructSeg2019_Task4_Lung_GTV -m 3d_lowres -f 0 1 2 3 4 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True --step 2





# Task49 play with gaussian (step). default is 2 = move by patch_size / 2. Let's set this to 1.5 to move by 2 * patch_size / 3
'''python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_fast_step1 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 1
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_fast_step1333 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 1.33333333
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_fast_step15 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 1.5
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task49_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task49_TestSet/predictions_fast_step2 -t Task49_StructSeg2019_Task1_HaN_OAR -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2


python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_fast_step1 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 1
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_fast_step1333333 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 1.333333333
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_fast_step15 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 1.5
python inference/predict_simple.py -i /media/fabian/DeepLearningData/simulated_Task50_TestSet/data -o /media/fabian/DeepLearningData/simulated_Task50_TestSet/predictions_fast_step2 -t Task50_StructSeg2019_Task2_Naso_GTV -m 3d_fullres -f 0 -tr nnUNetTrainerV2_2_noMirror --tta 0 --num_threads_preprocessing 4 --num_threads_nifti_save 6 --mode fast --all_in_gpu True -p nnUNetPlans_customClip --step 2
'''

