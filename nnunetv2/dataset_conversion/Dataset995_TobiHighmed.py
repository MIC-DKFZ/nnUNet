from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import shutil
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.paths import nnUNet_raw

if __name__ == '__main__':
    dataset_name = 'Dataset995_TobiHighmed'
    source_folder = '/home/isensee/drives/e132-projekte/Projects/2021_Tobi_Highmed/to_nnunet/nat'

    output_folder = join(nnUNet_raw, dataset_name)
    maybe_mkdir_p(output_folder)

    shutil.copytree(join(source_folder, 'imagesTr'), join(output_folder, 'imagesTr'))
    shutil.copytree(join(source_folder, 'labelsTr'), join(output_folder, 'labelsTr'))

    generate_dataset_json(output_folder, {0: 'MRI'}, {'tumor': 1, 'background': 0}, 10, '.nii.gz', dataset_name)
