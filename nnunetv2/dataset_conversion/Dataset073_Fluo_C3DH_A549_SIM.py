from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import nnunetv2.paths as paths
import tifffile
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


if __name__ == '__main__':
    """
    This is going to be my test dataset for working with tif as input and output images
    
    All we do here is copy the files and rename them. Not file conversions take place 
    """
    dataset_name = 'Dataset073_Fluo_C3DH_A549_SIM'

    imagestr = join(paths.nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(paths.nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(paths.nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    # we extract the downloaded train and test datasets to two separate folders and name them Fluo-C3DH-A549-SIM_train
    # and Fluo-C3DH-A549-SIM_test
    train_source = '/home/fabian/Downloads/Fluo-C3DH-A549-SIM_train'
    test_source = '/home/fabian/Downloads/Fluo-C3DH-A549-SIM_test'

    # with the old nnU-Net we had to convert all the files to nifti. This is no longer required. We can just copy the
    # tif files

    # tif is broken when it comes to spacing. No standards. Grr. So when we use tif nnU-Net expects a separate file
    # that specifies the spacing. This file needs to exist for EVERY training/test case to allow for different spacings
    # between files. Important! The spacing must align with the axes.
    # Here when we do print(tifffile.imread('IMAGE').shape) we get (29, 300, 350). The low resolution axis is the first.
    # The spacing on the website is griven in the wrong axis order. Great.
    spacing = (1, 0.126, 0.126)

    # train set
    for seq in ['01', '02']:
        images_dir = join(train_source, seq)
        seg_dir = join(train_source, seq + '_GT', 'SEG')
        # if we were to be super clean we would go by IDs but here we just trust the files are sorted the correct way.
        # Simpler filenames in the cell tracking challenge would be soooo nice.
        images = subfiles(images_dir, suffix='.tif', sort=True, join=False)
        segs = subfiles(seg_dir, suffix='.tif', sort=True, join=False)
        for i, (im, se) in enumerate(zip(images, segs)):
            target_name = f'{seq}_image_{i:03d}'
            # we still need the '_0000' suffix for images! Otherwise we would not be able to support multiple input
            # channels distributed over separate files
            shutil.copy(join(images_dir, im), join(imagestr, target_name + '_0000.tif'))
            # spacing file!
            save_json({'spacing': spacing}, join(imagestr, target_name + '.json'))
            shutil.copy(join(seg_dir, se), join(labelstr, target_name + '.tif'))
            # spacing file!
            save_json({'spacing': spacing}, join(labelstr, target_name + '.json'))

    # test set, same a strain just without the segmentations
    for seq in ['01', '02']:
        images_dir = join(test_source, seq)
        images = subfiles(images_dir, suffix='.tif', sort=True, join=False)
        for i, im in enumerate(images):
            target_name = f'{seq}_image_{i:03d}'
            shutil.copy(join(images_dir, im), join(imagests, target_name + '_0000.tif'))
            # spacing file!
            save_json({'spacing': spacing}, join(imagests, target_name + '.json'))

    # now we generate the dataset json
    generate_dataset_json(
        join(paths.nnUNet_raw, dataset_name),
        {0: 'fluorescence_microscopy'},
        {'background': 0, 'cell': 1},
        60,
        '.tif'
    )

    # custom split to ensure we are stratifying properly. This dataset only has 2 folds
    caseids = [i[:-4] for i in subfiles(labelstr, suffix='.tif', join=False)]
    splits = []
    splits.append(
        {'train': [i for i in caseids if i.startswith('01_')], 'val': [i for i in caseids if i.startswith('02_')]}
    )
    splits.append(
        {'train': [i for i in caseids if i.startswith('02_')], 'val': [i for i in caseids if i.startswith('01_')]}
    )
    save_json(splits, join(paths.nnUNet_preprocessed, dataset_name, 'splits_final.json'))
