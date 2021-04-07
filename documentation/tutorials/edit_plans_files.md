Changing the plans files grants you a lot of flexibility: You can depart from nnU-Net's default configuration and play 
with different U-Net topologies, batch sizes and patch sizes. It is a powerful tool!
To better understand the components describing the network topology in our plans files, please read section 6.2 
in the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf) 
(page 13) of our paper!
    
The goal of this tutorial is to demonstrate how to read and modify plans files and how to use them in your 
experiments. The file used here works with Task120 and requires you to have downloaded the dataset, run 
nnunet.dataset_conversion.Task120_Massachusetts_RoadSegm.py and then run nnUNet_plan_and_preprocess for it.

Note that this task is 2D only, but the same principles we use here can be easily extended to 3D and other tasks as well.

The output of `nnUNet_plan_and_preprocess` for this task looks like this:

    [{'batch_size': 2, 
    'num_pool_per_axis': [8, 8], 
    'patch_size': array([1280, 1024]), 
    'median_patient_size_in_voxels': array([   1, 1500, 1500]), 
    'current_spacing': array([999.,   1.,   1.]), 
    'original_spacing': array([999.,   1.,   1.]), 
    'pool_op_kernel_sizes': [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]], 
    'conv_kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], 
    'do_dummy_2D_data_aug': False}]

This is also essentially what is saved in the plans file under the key 'plans_per_stage'

For this task, nnU-Net intends to use a patch size of 1280x1024 and a U-Net architecture with 8 pooling 
operations per axis. Due to GPU memory constraints, the batch size is just 2.

Knowing the dataset we could hypothesize that a different approach might produce better results: The decision 
of whether a pixel belongs to 'road' or not does not depend on the large contextual information that the large 
patch size (and U-Net architecture) offer and could instead be made with more local information. Training with
a batch size of just 2 in a dataset with 800 training cases means that each batch contains only limited variability.
So one possible conclusion could be that smaller patches but larger batch sizes might result in a better 
segmentation outcome. Let's investigate (using the same GPU memory constraint, determined manually with trial 
and error!):

Variant 1: patch size 512x512, batch size 12
The following snippet makes the necessary adaptations to the plans file

```python
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir
task_name = 'Task120_MassRoadsSeg'

# if it breaks upon loading the plans file, make sure to run the Task120 dataset conversion and
# nnUNet_plan_and_preprocess first!
plans_fname = join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_plans_2D.pkl')
plans = load_pickle(plans_fname)
plans['plans_per_stage'][0]['batch_size'] = 12
plans['plans_per_stage'][0]['patch_size'] = np.array((512, 512))
plans['plans_per_stage'][0]['num_pool_per_axis'] = [7, 7]
# because we changed the num_pool_per_axis, we need to change conv_kernel_sizes and pool_op_kernel_sizes as well!
plans['plans_per_stage'][0]['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
plans['plans_per_stage'][0]['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
# for a network with num_pool_per_axis [7,7] the correct length of pool kernel sizes is 7 and the length of conv
# kernel sizes is 8! Note that you can also change these numbers if you believe it makes sense. A pool kernel size
# of 1 will result in no pooling along that axis, a kernel size of 3 will reduce the size of the feature map
# representations by factor 3 instead of 2.

# save the plans under a new plans name. Note that the new plans file must end with _plans_2D.pkl!
save_pickle(plans, join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_ps512_bs12_plans_2D.pkl'))
```


Variant 2: patch size 256x256, batch size 60

```python
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir
task_name = 'Task120_MassRoadsSeg'
plans_fname = join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_plans_2D.pkl')
plans = load_pickle(plans_fname)
plans['plans_per_stage'][0]['batch_size'] = 60
plans['plans_per_stage'][0]['patch_size'] = np.array((256, 256))
plans['plans_per_stage'][0]['num_pool_per_axis'] = [6, 6]
plans['plans_per_stage'][0]['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
plans['plans_per_stage'][0]['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
save_pickle(plans, join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_ps256_bs60_plans_2D.pkl'))
```

You can now use these custom plans files to train the networks and compare the results! Remeber that all nnUNet_* 
commands have the -h argument to display their arguments. nnUNet_train supports custom plans via the -p argument. 
Custom plans must be the prefix, so here this is everything except '_plans_2D.pkl':

Variant 1:
```bash
nnUNet_train 2d nnUNetTrainerV2 120 FOLD -p nnUNetPlansv2.1_ps512_bs12
```

Variant 2:
```bash
nnUNet_train 2d nnUNetTrainerV2 120 FOLD -p nnUNetPlansv2.1_ps256_bs60
```


Let all 5 folds run for each plans file (original and the two variants). To compare the results, you can make use of
nnUNet_determine_postprocessing to get the necessary metrics, for example:

```bash
nnUNet_determine_postprocessing -t 120 -tr nnUNetTrainerV2 -p nnUNetPlansv2.1_ps512_bs12 -m 2d
```

This will create a `cv_niftis_raw` and `cv_niftis_postprocessed` subfolder in the training output directory. In each
 of these folders is a summary.json file that you can open with a regular text editor. In this file, there are metrics 
 for each training example in the dataset representing the outcome of the 5-fold cross-validation. At the very bottom 
 of the file, the metrics are aggregated through averaging (field "mean") and this is what you should be using to 
 compare the experiments. I recommend using the non-postprocessed summary.json (located in `cv_niftis_raw`) for this 
 because determining the postprocessing may actually overfit to the training dataset. Here are the results I obtained:
 
Vanilla nnU-Net:    0.7720\
Variant 1: 0.7724\
Variant 2: 0.7734

The results are remarkable similar and I would not necessarily conclude that such a small improvement in Dice is a 
significant outcome. Nonetheless it was worth a try :-)

Despite the results shown here I would like to emphasize that modifying the plans file can be an extremely powerful 
tool to improve the performance of nnU-Net on some datasets. You never know until you try it.

**ADDITIONAL INFORMATION (READ THIS!)**

  - when working with 3d plans ('nnUNetPlansv2.1_plans_3D.pkl') the 3d_lowres and 3d_fullres stage will be encoded 
    in the same plans file. If len(plans['plans_per_stage']) == 2, then [0] is the 3d_lowres and [1] is the 
    3d_fullres variant. If len(plans['plans_per_stage']) == 1 then [0] will be 3d_fullres and 3d_cascade_fullres 
    (they use the same plans).
    
  - 'pool_op_kernel_sizes' together with determines 'patch_size' determines the size of the feature map 
    representations at the bottleneck. For Variant 1 & 2 presented here, the size of the feature map representation is
    
    `print(plans['plans_per_stage'][0]['patch_size'] / np.prod(plans['plans_per_stage'][0]['pool_op_kernel_sizes'], 0))`
    
    > [4., 4.]
    
    If you see a non-integer number here, your model will crash! Make sure these are always integers!
    nnU-Net will never create smaller bottlenecks than 4!

  - do not change the 'current_spacing' in the plans file! This will not work properly. To change the target spacing, 
  have a look at the [custom spacing](custom_spacing.md) tutorial.