from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import SimpleITK as sitk


if __name__ == '__main__':
    """
    
    """
    # extracted traiing.zip file is here
    base = '/home/isensee/Downloads/'
    target_dataset_id = 21
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_CTAAorta'

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    cases = subfiles(join(base, 'images'), join=False, prefix='subject')
    for case in cases:
        outname = case.replace('_CTA.mha', '')
        im = sitk.ReadImage(join(base, 'images', case))
        sitk.WriteImage(im, join(imagesTr, outname + '_0000.nii.gz'))

        seg = sitk.ReadImage(join(base, 'masks', case.replace('_CTA.mha', '_label.mha')))
        sitk.WriteImage(seg, join(labelsTr, outname + '.nii.gz'))

    labels = {
            "background": 0,
            "Zone_0": 1,
            "Innominate": 2,
            "Zone_1": 3,
            "Left_Common_Carotid": 4,
            "Zone_2": 5,
            "Left_Subclavian_Artery": 6,
            "Zone_3": 7,
            "Zone_4": 8,
            "Zone_5": 9,
            "Zone_6": 10,
            "Celiac_Artery": 11,
            "Zone_7": 12,
            "SMA": 13,
            "Zone_8": 14,
            "Right_Renal_Artery": 15,
            "Left_Renal_Artery": 16,
            "Zone_9": 17,
            "Zone_10_R_(Right_Common_Iliac_Artery)": 18,
            "Zone_10_L_(Left_Common_Iliac_Artery)": 19,
            "Right_Internal_Iliac_Artery_Dice_Score": 20,
            "Left_Internal_Iliac_Artery_Dice_Score": 21,
            "Zone_11_R_(Right_External_Iliac_Artery)": 22,
            "Zone_11_L_(Left_External_Iliac_Artery)": 23
        }


    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        {0: 'CTA'},
        labels,
        len(cases),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient',
        reference='https://aortaseg24.grand-challenge.org/',
        license='see ref'
    )