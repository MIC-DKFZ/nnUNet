import csv
from copy import deepcopy

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import SimpleITK as sitk

# label_code of the RibFrac info csv files. -1: it is a rib fracture, but its type could not be defined due to
# ambiguity, diagnosis difficulty, etc. (ignore it in the classification task)
RIBFRAC_TYPES = {
    1: 'displaced rib fracture',
    2: 'non-displaced rib fracture',
    3: 'buckle rib fracture',
    4: 'segmental rib fracture',
    -1: 'rib fracture of unknown type',
}
RIBFRAC_INFO_CSVS = ('ribfrac-train-info-1.csv', 'ribfrac-train-info-2.csv', 'ribfrac-val-info.csv')


def load_ribfrac_info(info_dir: str) -> dict:
    """
    Returns {public_id: {label_id: label_code}} for all fracture instances in the RibFrac info csv files
    (label_id 0 = background is skipped).
    """
    info = {}
    for f in RIBFRAC_INFO_CSVS:
        with open(join(info_dir, f)) as fh:
            for row in csv.DictReader(fh):
                if int(row['label_id']) != 0:
                    info.setdefault(row['public_id'], {})[int(row['label_id'])] = int(row['label_code'])
    return info


def write_fracture_types_csv(info: dict, output_file: str) -> None:
    with open(output_file, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['public_id', 'label_id', 'label_code', 'type'])
        for case in sorted(info, key=lambda c: int(c[len('RibFrac'):])):
            for label_id in sorted(info[case]):
                writer.writerow([case, label_id, info[case][label_id], RIBFRAC_TYPES[info[case][label_id]]])


def check_instances(seg_npy: np.ndarray, case: str, info: dict) -> None:
    """
    The RibFrac -label.nii.gz files hold one ID per fracture instance (label_id 1..n, 0 = background). The fracture
    type of each instance is only in the info csv files, there is no -1 in the label files.
    """
    assert seg_npy.min() >= 0 and seg_npy.max() <= 255, f'{case}: unexpected label values'
    ids = set(np.unique(seg_npy).tolist()) - {0}
    assert ids == set(info.get(case, {})), f'{case}: label ids {sorted(ids)} do not match the info csv'


def write_label(seg_npy: np.ndarray, reference: sitk.Image, output_file: str) -> None:
    out = sitk.GetImageFromArray(seg_npy.astype(np.uint8))
    out.CopyInformation(reference)
    sitk.WriteImage(out, output_file)


def migrate_existing_dataset015(info_dir: str, dataset_name: str = 'Dataset015_RibFrac') -> None:
    """
    One-time fix for a Dataset015 made with the previous version of this script. Its labelsTr hold the original
    fracture instance IDs (the old '-1 -> 5' mapping never applied because the label files contain no -1) while
    dataset.json declared fracture types. This moves labelsTr to labelsTr_instances, writes binary labelsTr
    (any fracture -> 1) and fracture_types.csv (label_id -> type per case). dataset.json is NOT touched, update it
    separately (labels: background 0, rib fracture 1).
    info_dir must contain the three RibFrac info csv files of the Zenodo release.
    """
    base_dir = join(nnUNet_raw, dataset_name)
    assert not isdir(join(base_dir, 'labelsTr_instances')), 'labelsTr_instances exists already, migrated before?'
    info = load_ribfrac_info(info_dir)
    files = nifti_files(join(base_dir, 'labelsTr'), join=False)
    # check everything before changing anything
    for f in files:
        check_instances(sitk.GetArrayFromImage(sitk.ReadImage(join(base_dir, 'labelsTr', f))), f[:-len('.nii.gz')], info)
    shutil.move(join(base_dir, 'labelsTr'), join(base_dir, 'labelsTr_instances'))
    maybe_mkdir_p(join(base_dir, 'labelsTr'))
    for f in files:
        seg_itk = sitk.ReadImage(join(base_dir, 'labelsTr_instances', f))
        write_label(sitk.GetArrayFromImage(seg_itk) > 0, seg_itk, join(base_dir, 'labelsTr', f))
    write_fracture_types_csv(info, join(base_dir, 'fracture_types.csv'))


if __name__ == '__main__':
    """
    Download RibFrac dataset. Links are at https://ribfrac.grand-challenge.org/
    Download everything. Part1, 2, validation and test, including the ribfrac-*-info.csv files
    Extract EVERYTHING into one folder so that all images and labels are in there. Don't worry they all have unique
    file names.

    Dataset015 gets binary fracture labels in labelsTr (rib fracture detection/segmentation), the fracture instances in
    labelsTr_instances and the type of each instance in fracture_types.csv. Most fractures (58%) have no type
    (label_code -1), so fracture type classification is left to datasets derived from labelsTr_instances.
    If you have a Dataset015 from the previous version of this script, run migrate_existing_dataset015(<folder with
    the info csv files>) instead of converting again.

    For RibSeg also download the dataset from https://github.com/M3DV/RibSeg
    (https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm/view?usp=sharing) and extract in to that same
    folder (seg only, files end with -rib-seg.nii.gz)
    """
    # extracted training.zip file is here
    base = '/home/isensee/Downloads/RibFrac_all'

    files = nifti_files(base, join=False)
    identifiers = np.unique([i.split('-')[0] for i in files])

    # RibFrac
    target_dataset_id = 15
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_RibFrac'

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    imagesTs = join(nnUNet_raw, target_dataset_name, 'imagesTs')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    labelsTr_instances = join(nnUNet_raw, target_dataset_name, 'labelsTr_instances')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(imagesTs)
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(labelsTr_instances)

    info = load_ribfrac_info(base)
    n_tr = 0
    for c in identifiers:
        print(c)
        img_file = join(base, c + '-image.nii.gz')
        seg_file = join(base, c + '-label.nii.gz')
        if not isfile(seg_file):
            # test case
            shutil.copy(img_file, join(imagesTs, c + '_0000.nii.gz'))
            continue
        n_tr += 1
        shutil.copy(img_file, join(imagesTr, c + '_0000.nii.gz'))

        # the label files hold fracture instance IDs, the types are in the info csv files
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)
        check_instances(seg_npy, c, info)
        write_label(seg_npy, seg_itk, join(labelsTr_instances, c + '.nii.gz'))
        write_label(seg_npy > 0, seg_itk, join(labelsTr, c + '.nii.gz'))
    write_fracture_types_csv(info, join(nnUNet_raw, target_dataset_name, 'fracture_types.csv'))

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        channel_names={0: 'CT'},
        labels={
            'background': 0,
            'rib fracture': 1,
        },
        num_training_cases=n_tr,
        file_ending='.nii.gz',
        dataset_name=target_dataset_name,
        reference='https://ribfrac.grand-challenge.org/',
        license='CC-BY-NC-4.0',
        converted_by='Fabian Isensee',
        note='labelsTr: binary rib fracture segmentation. labelsTr_instances: fracture instance IDs as in the RibFrac '
             'release; fracture_types.csv maps each instance (public_id, label_id) to its type (label_code -1 = type '
             'unknown).'
    )

    # RibSeg
    # overall I am not happy with the GT quality here. But eh what can I do

    target_dataset_name_ribfrac = deepcopy(target_dataset_name)
    target_dataset_id = 18
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_RibSeg'

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # the authors have a google shet where they highlight problems with their dataset:
    # https://docs.google.com/spreadsheets/d/1lz9liWPy8yHybKCdO3BCA9K76QH8a54XduiZS_9fK70/edit?gid=1416415020#gid=1416415020
    # we exclude the cases marked in red. They have unannotated ribs
    skip_identifiers = [
        'RibFrac452',
        'RibFrac485',
        'RibFrac490',
        'RibFrac471',
        'RibFrac462',
        'RibFrac487',
    ]

    n_tr = 0
    dataset = {}
    for c in identifiers:
        if c in skip_identifiers:
            continue
        print(c)
        tr_file = join('$nnUNet_raw', target_dataset_name_ribfrac, 'imagesTr', c + '_0000.nii.gz')
        ts_file = join('$nnUNet_raw', target_dataset_name_ribfrac, 'imagesTs', c + '_0000.nii.gz')
        if isfile(os.path.expandvars(tr_file)):
            img_file = tr_file
        elif isfile(os.path.expandvars(ts_file)):
            img_file = ts_file
        else:
            raise RuntimeError(f'Missing image file for identifier {identifiers}')
        seg_file = join(base, c + '-rib-seg.nii.gz')
        n_tr += 1
        shutil.copy(seg_file, join(labelsTr, c + '.nii.gz'))
        dataset[c] = {
            'images': [img_file],
            'label': join('labelsTr', c + '.nii.gz')
        }

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        channel_names={0: 'CT'},
        labels = {
            'background': 0,
            **{'rib%02.0d' % i: i for i in range(1, 25)}
        },
        num_training_cases=n_tr,
        file_ending='.nii.gz',
        dataset_name=target_dataset_name,
        reference='https://github.com/M3DV/RibSeg, https://ribfrac.grand-challenge.org/',
        dataset=dataset
    )
