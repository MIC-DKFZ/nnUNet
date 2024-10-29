from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

if __name__ == '__main__':
    """
    Download the dataset from huggingface:
    https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini#3--download-the-dataset
    
    IMPORTANT
    cases 5196-9262 currently do not have images, just the segmentation. This seems to be a mistake 
    """
    base = '/home/isensee/Downloads/AbdomenAtlas/uncompressed'
    target_dataset_id = 23
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_AbdomenAtlas1.1Mini'

    cases = subdirs(base, join=False, prefix='BDMAP')

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    for case in cases:
        if not isfile(join(base, case, 'ct.nii.gz')):
            print(f'Skipping case {case} due to missing image')
            continue
        shutil.copy(join(base, case, 'ct.nii.gz'), join(imagesTr, case + '_0000.nii.gz'))
        shutil.copy(join(base, case, 'combined_labels.nii.gz'), join(labelsTr, case + '.nii.gz'))

    class_map = {1: 'aorta', 2: 'gall_bladder', 3: 'kidney_left', 4: 'kidney_right', 5: 'liver',
                 6: 'pancreas', 7: 'postcava', 8: 'spleen', 9: 'stomach', 10: 'adrenal_gland_left',
                 11: 'adrenal_gland_right', 12: 'bladder', 13: 'celiac_trunk', 14: 'colon', 15: 'duodenum',
                 16: 'esophagus', 17: 'femur_left', 18: 'femur_right', 19: 'hepatic_vessel', 20: 'intestine',
                 21: 'lung_left', 22: 'lung_right', 23: 'portal_vein_and_splenic_vein',
                 24: 'prostate', 25: 'rectum'}
    labels = {
        j: i for i, j in class_map.items()
    }
    labels['background'] = 0

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        {0: 'CT'},
        labels,
        len(cases),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient',
        reference='https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini',
        license='Creative Commons Attribution Non Commercial Share Alike 4.0; see reference'
    )