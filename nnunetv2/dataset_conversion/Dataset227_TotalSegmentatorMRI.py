import SimpleITK
import nibabel
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw



if __name__ == '__main__':
    base = '/home/isensee/Downloads/TotalsegmentatorMRI_dataset_v100'
    cases = subdirs(base, join=False)

    target_dataset_id = 227
    target_dataset_name = f'Dataset{target_dataset_id:3.0f}_TotalSegmentatorMRI'

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # discover labels
    label_fnames = nifti_files(join(base, cases[0], 'segmentations'), join=False)
    label_dict = {i[:-7]: j + 1 for j, i in enumerate(label_fnames)}
    labelnames = list(label_dict.keys())

    for case in cases:
        img = nibabel.load(join(base, case, 'mri.nii.gz'))
        nibabel.save(img, join(imagesTr, case + '_0000.nii.gz'))

        seg_nib = nibabel.load(join(base, case, 'segmentations', labelnames[0] + '.nii.gz'))
        init_seg_npy = np.asanyarray(seg_nib.dataobj)
        init_seg_npy[init_seg_npy > 0] = label_dict[labelnames[0]]
        for labelname in labelnames[1:]:
            seg = nibabel.load(join(base, case, 'segmentations', labelname + '.nii.gz'))
            seg = np.asanyarray(seg.dataobj)
            init_seg_npy[seg > 0] = label_dict[labelname]
        out = nibabel.Nifti1Image(init_seg_npy, affine=seg_nib.affine, header=seg_nib.header)
        nibabel.save(out, join(labelsTr, case + '.nii.gz'))

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        {0: 'MRI'},  # this was a mistake we did at the beginning and we keep it like that here for consistency
        label_dict,
        len(cases),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient',
        release='1.0.0',
        reference='https://zenodo.org/records/11367005',
        license='see reference'
    )