import shutil

from batchgenerators.utilities.file_and_folder_operations import subdirs, isdir, join

from nnunetv2.paths import nnUNet_preprocessed

if __name__ == '__main__':

    mapping = {
        '2d': 'nnUNetPlans_2d',
        '3d_fullres': 'nnUNetPlans_3d_fullres',
        '3d_lowres': 'nnUNetPlans_3d_lowres',
        '2d_resencunet': 'nnUNetResEncUNetPlans_2d',
        '3d_fullres_resencunet': 'nnUNetResEncUNetPlans_3d_fullres',
        '3d_lowres_resencunet': 'nnUNetResEncUNetPlans_3d_lowres',
    }

    for d in subdirs(nnUNet_preprocessed):
        for src, trg in mapping.items():
            if isdir(join(d, src)):
                shutil.move(join(d, src), join(d, trg))
