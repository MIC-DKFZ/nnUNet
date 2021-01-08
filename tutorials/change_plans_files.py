#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.paths import preprocessing_output_dir


if __name__ == '__main__':
    """
    The goal of this tutorial is to demonstrate how to read and modify plans files and how to use them in your 
    experiments. The file used here works with Task120 and requires you to have run nnUNet_plan_and_preprocess for it.
    Note that this task is 2D only, but the same principles we use here can be applied to the other tasks as well
    """

    """
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
    So one possible conclusion could be that smaller patches but larger patch sizes might result in a better 
    segmentation outcome. Let's investigate (using the same GPU memory constraint, determined manually with trial 
    and error!):
    """

    """
    Variant 1: patch size 512x512, batch size 12
    The following snippet makes the necessary adaptations to the plans file
    """
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

    """
    Variant 2: patch size 256x256, batch size 48
    """
    plans = load_pickle(plans_fname)
    plans['plans_per_stage'][0]['batch_size'] = 60
    plans['plans_per_stage'][0]['patch_size'] = np.array((256, 256))
    plans['plans_per_stage'][0]['num_pool_per_axis'] = [6, 6]
    plans['plans_per_stage'][0]['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    plans['plans_per_stage'][0]['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    save_pickle(plans, join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_ps256_bs60_plans_2D.pkl'))

    """
    You can now use these custom plans files to train the networks and compare the results! Remeber that all nnUNet_* 
    commands have the -h argument to display their arguments. nnUNet_train supports custom plans via the -p argument. 
    Custom plans must be the prefix, so here this is everything except '_plans_2D.pkl':
    
    Variant 1:
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD -p nnUNetPlansv2.1_ps512_bs12
    
    Variant 2:
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD -p nnUNetPlansv2.1_ps256_bs60
    
    Let all 5 folds run for each plans file (original and the two variants). To compare the results, you can make use of
    nnUNet_determine_postprocessing to get the necessary metrics 
    
    (TO BE CONTINUED)
    """

    """
    ADDITIONAL INFORMATION (READ FOR 3D PLANS!)
    
      - when working with 3d plans ('nnUNetPlansv2.1_plans_3D.pkl') the 3d_lowres and 3d_fullres stage will be encoded 
        in the same plans file. If len(plans['plans_per_stage']) == 2, then [0] is the 3d_lowres and [1] is the 
        3d_fullres variant. If len(plans['plans_per_stage']) == 1 then [0] will be 3d_fullres and 3d_cascade_fullres 
        (they use the same plans).
        
      - 'pool_op_kernel_sizes' together with determines 'patch_size' determines the size of the feature map 
        representations at the bottleneck. For Variant 1 & 2 presented here, the size of the feature map representation is
        > print(plans['plans_per_stage'][0]['patch_size'] / np.prod(plans['plans_per_stage'][0]['pool_op_kernel_sizes'], 0))
        -> [4., 4.]
        If you see a non-integer number here, your model will crash! Make sure these are always integers!
        nnU-Net will never create smaller bottlenecks than 4!
        
    """