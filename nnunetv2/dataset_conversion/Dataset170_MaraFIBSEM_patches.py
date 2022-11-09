import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import tifffile
import SimpleITK as sitk

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

if __name__ == '__main__':
    source_dir = '/home/isensee/temp/annotated patches'

    dataset_name = 'Dataset170_MaraFIBSEM_patches'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    source_patches = subfiles(source_dir, suffix='.tif', join=False)
    for s in source_patches:
        image_file = join(source_dir, s)
        seg_file = join(source_dir, s[:-4] + '-labels.nrrd')

        # # try to map file name to spacing
        # if s.startswith('Helios_3x3x5'):
        #     spacing = (5, 3, 3)
        # elif s.startswith('Helis_4x5x5'):
        #     spacing = (4, 5, 5)
        # elif s.startswith('Scios_3x4x5'):
        #     spacing = (5, 4, 3)
        # elif s.startswith('Scios_4x5x5'):
        #     spacing = (4, 5, 5)
        # else:
        #     raise RuntimeError()
        spacing = (1, 1, 1)

        image = tifffile.imread(image_file)
        image_itk = sitk.GetImageFromArray(image)
        image_itk.SetSpacing(spacing)

        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
        assert np.sum(seg[1::5]) == 0
        assert np.sum(seg[2::5]) == 0
        assert np.sum(seg[3::5]) == 0
        assert np.sum(seg[4::5]) == 0
        seg[1::5] = 2
        seg[2::5] = 2
        seg[3::5] = 2
        seg[4::5] = 2

        seg_itk = sitk.GetImageFromArray(seg)
        seg_itk.SetSpacing(spacing)

        sitk.WriteImage(image_itk, join(imagestr, s[:-4] + '_0000.nii.gz'))
        sitk.WriteImage(seg_itk, join(labelstr, s[:-4] + '.nii.gz'))

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'noNorm'}, {'background': 0, 'membrane': 1, 'ignore': 2},
                          num_training_cases=28, file_ending='.nii.gz', dataset_name=dataset_name)

    # custom stratified split
    split = []
    prefixes = ['Helios_3x3x5', 'Helios_4x5x5', 'Scios_3x4x5nm', 'Scios_4x5x5nm']
    for f in range(4):
        val_prefixes = prefixes[f::4]
        tr_prefixes = [i for i in prefixes if i not in val_prefixes]
        split.append({'train': [], 'val': []})
        split[-1]['train'] = [i[:-4] for i in source_patches if any([i.startswith(j) for j in tr_prefixes])]
        split[-1]['val'] = [i[:-4] for i in source_patches if any([i.startswith(j) for j in val_prefixes])]
    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name))
    save_json(split, join(nnUNet_preprocessed, dataset_name, 'splits_final.json'))
