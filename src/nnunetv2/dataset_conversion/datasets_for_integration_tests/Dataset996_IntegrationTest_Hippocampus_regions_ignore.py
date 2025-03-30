import SimpleITK as sitk
import shutil

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json, save_json, nifti_files

from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def sparsify_segmentation(seg: np.ndarray, label_manager: LabelManager, percent_of_slices: float) -> np.ndarray:
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label
        x, y, z = seg.shape
        # x
        num_slices = max(1, round(x * percent_of_slices))
        selected_slices = np.random.choice(x, num_slices, replace=False)
        seg_new[selected_slices] = seg[selected_slices]
        # y
        num_slices = max(1, round(y * percent_of_slices))
        selected_slices = np.random.choice(y, num_slices, replace=False)
        seg_new[:, selected_slices] = seg[:, selected_slices]
        # z
        num_slices = max(1, round(z * percent_of_slices))
        selected_slices = np.random.choice(z, num_slices, replace=False)
        seg_new[:, :, selected_slices] = seg[:, :, selected_slices]
        return seg_new


if __name__ == '__main__':
    dataset_name = 'IntegrationTest_Hippocampus_regions_ignore'
    dataset_id = 996
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name}"

    try:
        existing_dataset_name = maybe_convert_to_dataset_name(dataset_id)
        if existing_dataset_name != dataset_name:
            raise FileExistsError(f"A different dataset with id {dataset_id} already exists :-(: {existing_dataset_name}. If "
                               f"you intent to delete it, remember to also remove it in nnUNet_preprocessed and "
                               f"nnUNet_results!")
    except RuntimeError:
        pass

    if isdir(join(nnUNet_raw, dataset_name)):
        shutil.rmtree(join(nnUNet_raw, dataset_name))

    source_dataset = maybe_convert_to_dataset_name(4)
    shutil.copytree(join(nnUNet_raw, source_dataset), join(nnUNet_raw, dataset_name))

    # additionally optimize entire hippocampus region, remove Posterior
    dj = load_json(join(nnUNet_raw, dataset_name, 'dataset.json'))
    dj['labels'] = {
        'background': 0,
        'hippocampus': (1, 2),
        'anterior': 1,
        'ignore': 3
    }
    dj['regions_class_order'] = (2, 1)
    save_json(dj, join(nnUNet_raw, dataset_name, 'dataset.json'), sort_keys=False)

    # now add ignore label to segmentation images
    np.random.seed(1234)
    lm = LabelManager(label_dict=dj['labels'], regions_class_order=dj.get('regions_class_order'))

    segs = nifti_files(join(nnUNet_raw, dataset_name, 'labelsTr'))
    for s in segs:
        seg_itk = sitk.ReadImage(s)
        seg_npy = sitk.GetArrayFromImage(seg_itk)
        seg_npy = sparsify_segmentation(seg_npy, lm, 0.1 / 3)
        seg_itk_new = sitk.GetImageFromArray(seg_npy)
        seg_itk_new.CopyInformation(seg_itk)
        sitk.WriteImage(seg_itk_new, s)

