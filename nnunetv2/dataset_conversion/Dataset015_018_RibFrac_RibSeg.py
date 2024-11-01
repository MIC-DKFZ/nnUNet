from copy import deepcopy

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import SimpleITK as sitk

if __name__ == '__main__':
    """
    Download RibFrac dataset. Links are at https://ribfrac.grand-challenge.org/
    Download everything. Part1, 2, validation and test
    Extract EVERYTHING into one folder so that all images and labels are in there. Don't worry they all have unique 
    file names.
    
    For RibSeg also download the dataset from https://github.com/M3DV/RibSeg 
    (https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm/view?usp=sharing) and extract in to that same 
    folder (seg only, files end with -rib-seg.nii.gz)
    """
    # extracted traiing.zip file is here
    base = '/home/isensee/Downloads/RibFrac_all'

    files = nifti_files(base, join=False)
    identifiers = np.unique([i.split('-')[0] for i in files])

    # RibFrac
    target_dataset_id = 15
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_RibFrac'

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    imagesTs = join(nnUNet_raw, target_dataset_name, 'imagesTs')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(imagesTs)
    maybe_mkdir_p(labelsTr)

    n_tr = 0
    for c in identifiers:
        print(c)
        img_file = join(base, c + '-image.nii.gz')
        seg_file = join(base, c + '-label.nii.gz')
        if not isfile(seg_file):
            # test case
            shutil.copy(img_file, join(imagesTs, c + '_0000.nii.gz'))
            continue
        n_tr += 1
        shutil.copy(img_file, join(imagesTr, c + '_0000.nii.gz'))

        # we must open seg and map -1 to 5
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)
        seg_npy[seg_npy == -1] = 5
        seg_itk_out = sitk.GetImageFromArray(seg_npy.astype(np.uint8))
        seg_itk_out.SetSpacing(seg_itk.GetSpacing())
        seg_itk_out.SetDirection(seg_itk.GetDirection())
        seg_itk_out.SetOrigin(seg_itk.GetOrigin())
        sitk.WriteImage(seg_itk_out, join(labelsTr, c + '.nii.gz'))

    # - 0: it is background
    # - 1: it is a displaced rib fracture
    # - 2: it is a non-displaced rib fracture
    # - 3: it is a buckle rib fracture
    # - 4: it is a segmental rib fracture
    # - -1: it is a rib fracture,  but we could not define its type due to
    #   ambiguity, diagnosis difficulty, etc. Ignore it in the
    #   classification task.

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        channel_names={0: 'CT'},
        labels = {
            'background': 0,
            'fracture': (1, 2, 3, 4, 5),
            'displaced rib fracture': 1,
            'non-displaced rib fracture': 2,
            'buckle rib fracture': 3,
            'segmental rib fracture': 4,
        },
        num_training_cases=n_tr,
        file_ending='.nii.gz',
        regions_class_order=(5, 1, 2, 3, 4),
        dataset_name=target_dataset_name,
        reference='https://ribfrac.grand-challenge.org/'
    )

    # RibSeg
    # overall I am not happy with the GT quality here. But eh what can I do

    target_dataset_name_ribfrac = deepcopy(target_dataset_name)
    target_dataset_id = 18
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_RibSeg'

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # the authors have a google shet where they highlight problems with their dataset:
    # https://docs.google.com/spreadsheets/d/1lz9liWPy8yHybKCdO3BCA9K76QH8a54XduiZS_9fK70/edit?gid=1416415020#gid=1416415020
    # we exclude the cases marked in red. They have unannotated ribs
    skip_identifiers = [
        'RibFrac452',
        'RibFrac485',
        'RibFrac490',
        'RibFrac471',
        'RibFrac462',
        'RibFrac487',
    ]

    n_tr = 0
    dataset = {}
    for c in identifiers:
        if c in skip_identifiers:
            continue
        print(c)
        tr_file = join('$nnUNet_raw', target_dataset_name_ribfrac, 'imagesTr', c + '_0000.nii.gz')
        ts_file = join('$nnUNet_raw', target_dataset_name_ribfrac, 'imagesTs', c + '_0000.nii.gz')
        if isfile(os.path.expandvars(tr_file)):
            img_file = tr_file
        elif isfile(os.path.expandvars(ts_file)):
            img_file = ts_file
        else:
            raise RuntimeError(f'Missing image file for identifier {identifiers}')
        seg_file = join(base, c + '-rib-seg.nii.gz')
        n_tr += 1
        shutil.copy(seg_file, join(labelsTr, c + '.nii.gz'))
        dataset[c] = {
            'images': [img_file],
            'label': join('labelsTr', c + '.nii.gz')
        }

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        channel_names={0: 'CT'},
        labels = {
            'background': 0,
            **{'rib%02.0d' % i: i for i in range(1, 25)}
        },
        num_training_cases=n_tr,
        file_ending='.nii.gz',
        dataset_name=target_dataset_name,
        reference='https://github.com/M3DV/RibSeg, https://ribfrac.grand-challenge.org/',
        dataset=dataset
    )
