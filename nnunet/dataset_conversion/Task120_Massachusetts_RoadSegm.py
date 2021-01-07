import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

if __name__ == '__main__':
    """
    nnU-Net was originally built for 3D images. It is also strongest when applied to 3D segmentation problems because a 
    large proportion of its design choices were built with 3D in mind. Also note that many 2D segmentation problems, 
    especially in the non-biomedical domain, may benefit from pretrained network architectures which nnU-Net does not
    support.
    Still, there is certainly a need for an out of the box segmentation solution for 2D segmentation problems. And 
    also on 2D segmentation tasks nnU-Net cam perform extremely well! We have, for example, won a 2D task in the cell 
    tracking challenge with nnU-Net (see our Nature Methods paper) and we have also successfully applied nnU-Net to 
    histopathological segmentation problems. 
    Working with 2D data in nnU-Net requires a small workaround in the creation of the dataset. Essentially, all images 
    must be converted to pseudo 3D images (so an image with shape (X, Y) needs to be converted to an image with shape 
    (1, X, Y). The resulting image must be saved in nifti format. Hereby it is important to set the spacing of the 
    first axis (the one with shape 1) to a value larger than the others. If you are working with niftis anyways, then 
    doing this should be easy for you. This example here is intended for demonstrating how nnU-Net can be used with 
    'regular' 2D images. We selected the massachusetts road segmentation dataset for this because it can be obtained 
    easily, it comes with a good amount of training cases but is still not too large to be difficult to handle.
    """

    # download dataset from https://www.kaggle.com/insaff/massachusetts-roads-dataset
    # extract the zip file, then set the following path according to your system:
    base = '/media/fabian/data/road_segmentation_ideal'
    # this folder should have the training and testing subfolders

    # now start the conversion to nnU-Net:
    task_name = 'Task120_MassRoadsSeg'
    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    labels_dir_tr = join(base, 'training', 'output')
    images_dir_tr = join(base, 'training', 'input')
    training_cases = subfiles(labels_dir_tr, suffix='.png', join=False)
    for t in training_cases:
        unique_name = t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_segmentation_file = join(labels_dir_tr, t)
        input_image_file = join(images_dir_tr, t)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 255).astype(int))

    # now do the same for the test set
    labels_dir_ts = join(base, 'testing', 'output')
    images_dir_ts = join(base, 'testing', 'input')
    testing_cases = subfiles(labels_dir_ts, suffix='.png', join=False)
    for ts in testing_cases:
        unique_name = ts[:-4]
        input_segmentation_file = join(labels_dir_ts, ts)
        input_image_file = join(images_dir_ts, ts)

        output_image_file = join(target_imagesTs, unique_name)
        output_seg_file = join(target_labelsTs, unique_name)

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 255).astype(int))

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Red', 'Green', 'Blue'),
                          labels={1: 'street'}, dataset_name=task_name, license='hands off!')

    """
    once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the 
    `nnUNet_plan_and_preprocess` command like this:
    
    > nnUNet_plan_and_preprocess -t 120 -pl3d None
    
    once that is completed, you can run the trainings as follows:
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD
    
    (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)
    
    there is no need to run nnUNet_find_best_configuration because there is only one model to shoose from.
    Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    `nnUNet_determine_postprocessing` command
    """

    #############################################################################################################

    """
    THIS PART IS OPTIONAL! It requires you to have run nnUNet_plan_and_preprocess.
    This section is purely here to demonstrate how you can manually edit the plans files to approach 
    a segmentation problem with a different patch size, batch size and U-Net architecture
    
    The output of `nnUNet_plan_and_preprocess` for this task looks like this (shortened!):
    
    [{'batch_size': 2, 'num_pool_per_axis': [8, 8], 'patch_size': array([1280, 1024]), ...]
    
    This means that nnU-Net intends to use a patch size of 1280x1024 and a U-Net architecture with 8 pooling 
    operations per axis. Due to GPU memory constraints, the batch size is just 2.
        
    Knowing the dataset we could hypothesize that a different approach might produce better results: The decision 
    of whether a pixel belongs to 'road' or not does not depent on the large contextual information that the large 
    patch size (and U-Net architecture) offer and could instead be made with more local information. Also, training with
    a batch size of just 2 in a dataset with 800 training cases means that each batch contains only limited variability.
    So one possible conclusion could be that smaller patches but larger patch sizes might result in a better 
    segmentation outcome. Let's investigate (using the same GPU memory constraint!):
    """

    """
    Variant 1: patch size 512x512, batch size 12
    The following snippet makes the necessary adaptations to the plans file
    """
    plans_fname = join(preprocessing_output_dir, task_name, 'nnUNetPlansv2.1_plans_2D.pkl')
    plans = load_pickle(plans_fname)
    plans['plans_per_stage'][0]['batch_size'] = 12
    plans['plans_per_stage'][0]['patch_size'] = np.array((512, 512))
    plans['plans_per_stage'][0]['num_pool_per_axis'] = [7, 7]
    # because we changed the num_pool_per_axis, we need to change conv_kernel_sizes and pool_op_kernel_sizes as well!
    plans['plans_per_stage'][0]['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    plans['plans_per_stage'][0]['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    # for a network with num_pool_per_axis [7,7] the correct length of pool kernel sizes is 7 and the length of conv
    # kernel sizes is 8! Note that you can also change these nubers if you believe it makes sense. A pool kernel size
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
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD -p nnUNetPlansv2.1_ps256_bs48
    
    Let all 5 folds run for each plans file (original and the two variants). To compare the results, you can make use of
    nnUNet_determine_postprocessing to get the necessary metrics 
    
    (TO BE CONTINUED)
    """


    """
    we could also add another experiment planner for downsampled images
    """