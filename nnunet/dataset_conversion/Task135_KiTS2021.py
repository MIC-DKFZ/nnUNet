from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
import shutil

from nnunet.paths import nnUNet_raw_data
from nnunet.dataset_conversion.utils import generate_dataset_json

if __name__ == '__main__':
    # this is the data folder from the kits21 github repository, see https://github.com/neheller/kits21
    kits_data_dir = '/home/fabian/git_repos/kits21/kits21/data'

    # This script uses the majority voted segmentation as ground truth
    kits_segmentation_filename = 'aggregated_MAJ_seg.nii.gz'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 135
    task_name = "KiTS2021"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(kits_data_dir, prefix='case_', join=False)
    for c in case_ids:
        if isfile(join(kits_data_dir, c, kits_segmentation_filename)):
            shutil.copy(join(kits_data_dir, c, kits_segmentation_filename), join(labelstr, c + '.nii.gz'))
            shutil.copy(join(kits_data_dir, c, 'imaging.nii.gz'), join(imagestr, c + '_0000.nii.gz'))

    generate_dataset_json(join(out_base, 'dataset.json'),
                          imagestr,
                          None,
                          ('CT',),
                          {
                              0: 'background',
                              1: "kidney",
                              2: "tumor",
                              3: "cyst",
                          },
                          task_name,
                          license='see https://kits21.kits-challenge.org/participate#download-block',
                          dataset_description='see https://kits21.kits-challenge.org/',
                          dataset_reference='https://www.sciencedirect.com/science/article/abs/pii/S1361841520301857, '
                                            'https://kits21.kits-challenge.org/',
                          dataset_release='0')
