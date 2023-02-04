# assumes you are in the nnunet repo!

# prepare raw datasets
python nnunetv2/dataset_conversion/datasets_for_integration_tests/Dataset999_IntegrationTest_Hippocampus.py
python nnunetv2/dataset_conversion/datasets_for_integration_tests/Dataset998_IntegrationTest_Hippocampus_ignore.py
python nnunetv2/dataset_conversion/datasets_for_integration_tests/Dataset997_IntegrationTest_Hippocampus_regions.py
python nnunetv2/dataset_conversion/datasets_for_integration_tests/Dataset996_IntegrationTest_Hippocampus_regions_ignore.py

# now run experiment planning without preprocessing
nnUNetv2_plan_and_preprocess -d 996 997 998 999 --no_pp

# now add 3d lowres and cascade
python nnunetv2/tests/integration_tests/add_lowres_and_cascade.py -d 996 997 998 999

# now preprocess everything
nnUNetv2_preprocess -d 996 997 998 999 -c 2d 3d_lowres 3d_fullres -np 8 8 8 8  # no need to preprocess cascade as its the same data as 3d_fullres

# done