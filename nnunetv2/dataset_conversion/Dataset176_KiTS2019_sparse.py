from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.dataset_conversion.Dataset179_KiTS2019_sparserer_blobs import compute_labeled_fractions_folder
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import SimpleITK as sitk
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def set_all_but_every_nth_slice_to_ignore(seg: np.ndarray, ignore_label: int, every_nth_slice: int = 10) -> np.ndarray:
    seg_new = np.ones_like(seg, dtype=np.uint8) * ignore_label
    seg_new[:, :, ::every_nth_slice] = seg[:, :, ::every_nth_slice]
    return seg_new


def load_set_all_but_every_nth_slice_to_ignore_save(source_img: str, target_img: str, ignore_label: int,
                                                    every_nth_slice: int):
    seg_itk = sitk.ReadImage(source_img)
    seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
    seg = set_all_but_every_nth_slice_to_ignore(seg, ignore_label, every_nth_slice)
    seg = sitk.GetImageFromArray(seg)
    seg.SetSpacing(seg_itk.GetSpacing())
    seg.SetDirection(seg_itk.GetDirection())
    seg.SetOrigin(seg_itk.GetOrigin())
    sitk.WriteImage(seg, target_img)


if __name__ == '__main__':
    """
    This dataset is a dummy to test different strategies on how to deal with ignore label . We need to test whether we 
    need adaptations to our hyperparameters depending on the amount and nature of annotations.
    
    This dataset only has every 10th slice annotated. All others are ignore label.
    See also Dataset 177
    """
    source_dataset_name = maybe_convert_to_dataset_name(64)
    dataset_name = 'Dataset176_KiTS2019_sparse'

    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(labelstr)
    # we can just copy the images
    shutil.copytree(join(nnUNet_raw, source_dataset_name, 'imagesTr'), join(nnUNet_raw, dataset_name, 'imagesTr'))

    ignore_label = 3  # 0 bg, 1 kidney, 2 tumor
    every_nth_slice = 10

    source_labels = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)

    p = Pool(16)
    r = []
    for s in source_labels:
        r.append(
            p.starmap_async(load_set_all_but_every_nth_slice_to_ignore_save,
                            ((
                                 join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                                 join(nnUNet_raw, dataset_name, 'labelsTr', s),
                                 ignore_label,
                                 every_nth_slice
                             ),))
        )
    _ = [i.get() for i in r]
    p.close()
    p.join()
    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'CT'},
                          {'background': 0, 'kidney': 1, 'tumor': 2, 'ignore': ignore_label},
                          210, '.nii.gz')
    # compute class fractions
    print(compute_labeled_fractions_folder(join(nnUNet_raw, maybe_convert_to_dataset_name(64), 'labelsTr'),
                                           join(nnUNet_raw, maybe_convert_to_dataset_name(176), 'labelsTr'),
                                            labels=(0, 1, 2),
                                           ignore_label=3, num_processes=16
                                           ))

    # add folds to splits_final.json of d64 that reflect annotated dataset percentage
    splits_final_file = join(nnUNet_preprocessed, 'Dataset064_KiTS_labelsFixed/splits_final.json')
    splits = load_json(splits_final_file)
    num_cases = len(splits[0]['train']) + len(splits[0]['val'])
    # add 5 folds for 10%. Keep val set of fold 0
    assert len(splits) == 5
    for n in range(5):
        splits.append(
            {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases / 10))),
             'val': splits[0]['val']}
        )
    # add 5 folds with 3%
    for n in range(5):
        splits.append(
            {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases / 33))),
             'val': splits[0]['val']}
        )
    # add 5 folds with .5%
    for n in range(5):
        splits.append(
            {'train': list(np.random.choice(splits[0]['train'], size=max(1, round(num_cases / 200)))),
             'val': splits[0]['val']}
        )
    # save
    save_json(splits, splits_final_file)


