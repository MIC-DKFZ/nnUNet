import SimpleITK
import h5py
import numpy as np
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
from batchgenerators.utilities.file_and_folder_operations import *
import scipy as sp
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

if __name__ == '__main__':
    source = '/home/fabian/Downloads/CT'
    npy_files = subfiles(source, suffix='.npy', join=False)

    dataset_name = 'Dataset990_Daniel'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    for n in npy_files:
        data = np.load(join(source, n))
        prob = np.array(h5py.File(join(source, n[:-4] + '_Probabilities.h5'))['exported_data'])
        prob_smooth = np.copy(prob)
        sigma = 5
        for i in range(prob.shape[3]):
            prob_smooth[:, :, :, i] = sp.ndimage.gaussian_filter(prob[:, :, :, i], sigma)

        l_storage, l_embryo, l_background = [0, 1, 2]

        # label with higherst probabilitiy per voxel:
        seg = np.argmax(prob, axis=3)
        seg_smooth = np.argmax(prob_smooth, axis=3)
        seg_embryo = remove_all_but_largest_component(seg_smooth == l_embryo)

        data_itk = SimpleITK.GetImageFromArray(data)
        SimpleITK.WriteImage(data_itk, join(imagestr, n[:-4] + '_0000.nii.gz'))
        seg_itk = SimpleITK.GetImageFromArray(seg_embryo.astype(np.uint8))
        SimpleITK.WriteImage(seg_itk, join(labelstr, n[:-4] + '.nii.gz'))

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'stuff'}, {'background': 0, 'random_1': 1, 'random_2': 2},
                          num_training_cases=len(npy_files), file_ending='.nii.gz')
