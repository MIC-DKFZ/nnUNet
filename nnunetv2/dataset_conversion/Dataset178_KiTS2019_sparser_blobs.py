import shutil
from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.Dataset177_KiTS2019_sparse_blobs import load_simulate_annotated_spheres_save
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

if __name__ == '__main__':
    """
    like 177 but with less fg sampling
    """
    source_dataset_name = maybe_convert_to_dataset_name(64)
    dataset_name = 'Dataset178_KiTS2019_sparser_blobs'

    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(labelstr)
    # we can just copy the images
    shutil.copytree(join(nnUNet_raw, source_dataset_name, 'imagesTr'), join(nnUNet_raw, dataset_name, 'imagesTr'))

    ignore_label = 3  # 0 bg, 1 kidney, 2 tumor
    num_spheres_random = 15
    num_spheres_fg = 3
    sphere_size = (15, 50)
    np.random.seed(12345)

    source_labels = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)

    p = Pool(16)
    r = []
    for s in source_labels:
        r.append(
            p.starmap_async(load_simulate_annotated_spheres_save,
                            ((
                                 join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                                 join(nnUNet_raw, dataset_name, 'labelsTr', s),
                                 num_spheres_random,
                                 num_spheres_fg,
                                 sphere_size,
                                 ignore_label,
                             ),))
        )
    _ = [i.get() for i in r]
    p.close()
    p.join()
    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'CT'},
                          {'background': 0, 'kidney': 1, 'tumor': 2, 'ignore': ignore_label},
                          210, '.nii.gz')