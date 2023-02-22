# Preface

I am just a mortal with many tasks and limited time. Aint nobody got time for unittests.

HOWEVER, at least some integration tests should be performed testing nnU-Net from start to finish.

# Introduction - What the heck is happening?
This test covers all possible labeling scenarios (standard labels, regions, ignore labels and regions with 
ignore labels). It runs the entire nnU-Net pipeline from start to finish:

- fingerprint extraction
- experiment planning
- preprocessing
- train all 4 configurations (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres) as 5-fold CV
- automatically find the best model or ensemble
- determine the postprocessing used for this
- predict some test set
- apply postprocessing to the test set

To speed things up, we do the following:
- pick Dataset004_Hipocampus because it is quadratisch praktisch gut. MNIST of medical image segmentation
- by default this dataset does not have 3d_lowres or cascade. We just manually add them (cool new feature, eh?). See `add_lowres_and_cascade.py` to learn more! 
- we use nnUNetTrainer_5epochs for a short training

# How to run it?

Set your pwd to be the nnunet repo folder (the one where the `nnunetv2` folder and the `setup.py` are located!)

Now generate the 4 dummy datasets (ids 996, 997, 998, 999) from dataset 4. This will crash if you don't have Dataset004!
```commandline
bash nnunetv2/tests/integration_tests/prepare_integration_tests.sh 
```

Now you can run the integration test for each of the datasets:
```commandline
bash nnunetv2/tests/integration_tests/run_integration_test.sh DATSET_ID
```
use DATSET_ID 996, 997, 998 and 999. You can run these independently on different GPUs/systems to speed things up. 
This will take i dunno like 10-30 Minutes!?

Also run 
```commandline
bash nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh DATSET_ID
```
to verify DDP is working (needs 2 GPUs!)

# How to check if the test was successful?
If I was not as lazy as I am I would have programmed some automatism that checks if Dice scores etc are in an acceptable range.
So you need to do the following:
1) check that none of your runs crashed (duh)
2) for each run, navigate to `nnUNet_results/DATASET_NAME` and take a look at the `inference_information.json` file. 
Does it make sense? If so: NICE!

Once the integration test is completed you can delete all the temporary files associated with it by running:

```commandline
python nnunetv2/tests/integration_tests/cleanup_integration_test.py
```