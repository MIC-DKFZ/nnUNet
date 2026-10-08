"""
QUADRA_HC: total-body [18F]FDG-PET/CT of 48 healthy controls (test/retest), Medical University of Vienna.

Download the raw data (CC BY 4.0, no registration needed):
    Zenodo record 16686025, version 1.1.1: https://zenodo.org/records/16686025
    Single file QUADRA_HC.zip (20,184,928,484 bytes, md5 5b37cd936988a023c67fe8f85d634d41):
    https://zenodo.org/records/16686025/files/QUADRA_HC.zip?download=1
    Zenodo downloads can be slow or break off; curl -C - resumes, parallel range requests are faster.
Extract it and pass the extracted QUADRA_HC folder (contains 'Imaging Data') as input_folder:
    python Dataset239_QUADRA_HC.py /path/to/QUADRA_HC [-o output_base] [-np 12]

The release ships one segmentation file per MOOSE model and case (automatic MOOSE predictions). We merge five of them
(Digestive, Organs, Muscles, Ribs, Vertebrae) into one label map with FIXED label IDs. Do not renumber the values that
happen to be present in a case: MOOSE leaves gaps for absent classes (e.g. vertebra L6 is absent in most cases), and
renumbering shifts every later label (this is what went wrong in the first conversion of this dataset).

Only the CT is used. The PET (PT-SUV) is not on the CT grid. Body-Composition, Cardiac and Peripheral-Bones are not used.
"""
import argparse
import shutil
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, subdirs, isfile

from nnunetv2.paths import nnUNet_raw

DATASET_NAME = 'Dataset239_QUADRA_HC'

# (MOOSE file, [(MOOSE label id, our label name), ...]) in merge order; target IDs are assigned consecutively from 1.
# MOOSE label definitions: https://github.com/ENHANCE-PET/MOOSE (README, clin_ct_* models).
# Where files overlap (median about 400, at most about 2,100 voxels per case) the later file wins.
MOOSE_MAPPING = [
    ('Digestive', [(1, 'large intestine'), (2, 'duodenum'), (3, 'oesophagus'), (4, 'jejunum and ileum')]),
    ('Organs', [(1, 'left adrenal gland'), (2, 'right adrenal gland'), (3, 'bladder'), (4, 'brain'), (5, 'gallbladder'),
                (6, 'kidney left'), (7, 'kidney right'), (8, 'liver'), (9, 'lung left lower lobe'),
                (10, 'lung right lower lobe'), (11, 'lung right middle lobe'), (12, 'lung left upper lobe'),
                (13, 'lung right upper lobe'), (14, 'pancreas'), (15, 'spleen'), (16, 'stomach'),
                (17, 'thyroid gland left'), (18, 'thyroid gland right'), (19, 'trachea')]),
    ('Muscles', [(1, 'longissimus thoracis left'), (2, 'longissimus thoracis right'), (3, 'glutes maximus left'),
                 (4, 'glutes maximus right'), (5, 'glutes medius left'), (6, 'glutes medius right'),
                 (7, 'glutes minimus left'), (8, 'glutes minimus right'), (9, 'iliopsoas left'), (10, 'iliopsoas right')]),
    # MOOSE has 13 ribs per side (left 1-13, right 14-26, sternum 27); the 13th ribs (13, 26) never occur in this release
    ('Ribs', [(i, f'left {i}. rib') for i in range(1, 13)] +
             [(13 + i, f'right {i}. rib') for i in range(1, 13)] +
             [(27, 'sternum')]),
    ('Vertebrae', [(i + 1, f'vertebrae {v}') for i, v in enumerate(
        [f'C{j}' for j in range(1, 8)] + [f'T{j}' for j in range(1, 13)] + [f'L{j}' for j in range(1, 7)])] +
                  [(26, 'left hip bone'), (27, 'right hip bone'), (28, 'sacrum')]),
]


def build_tables():
    labels = {'background': 0}
    luts = {}
    next_id = 1
    for model, entries in MOOSE_MAPPING:
        lut = np.full(256, -1, dtype=np.int16)  # -1 = value not expected in this file
        lut[0] = 0
        for moose_id, name in entries:
            lut[moose_id] = next_id
            labels[name] = next_id
            next_id += 1
        luts[model] = lut
    return labels, luts


def convert_case(source_dir: str, case: str, imagesTr: str, labelsTr: str, luts: dict) -> str:
    subject, session = case.rsplit('_', 1)
    case_dir = join(source_dir, 'Imaging Data', subject, session)
    shutil.copy(join(case_dir, f'{case}_CT-AC.nii.gz'), join(imagesTr, f'{case}_0000.nii.gz'))

    ref = None
    merged = None
    for model, _ in MOOSE_MAPPING:
        img = nib.load(join(case_dir, 'Segmentations', f'{case}_{model}.nii.gz'))
        seg = np.asarray(img.dataobj)
        if ref is None:
            ref = img
            merged = np.zeros(seg.shape, dtype=np.uint8)
        else:
            assert seg.shape == merged.shape and np.allclose(img.affine, ref.affine, atol=1e-4), \
                f'{case} {model}: segmentation grid differs from {MOOSE_MAPPING[0][0]}'
        mapped = luts[model][seg.astype(np.int64)]
        if (mapped < 0).any():
            raise RuntimeError(f'{case} {model}: unexpected label values {np.unique(seg[mapped < 0])}')
        mask = mapped > 0
        merged[mask] = mapped[mask]

    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    out = nib.Nifti1Image(merged, ref.affine, header)
    out.set_qform(ref.get_qform(), int(ref.header['qform_code']))
    out.set_sform(ref.get_sform(), int(ref.header['sform_code']))
    out.header.set_slope_inter(1, 0)
    nib.save(out, join(labelsTr, f'{case}.nii.gz'))
    return case


def convert(source_dir: str, output_base: str, num_processes: int = 12):
    out_dir = join(output_base, DATASET_NAME)
    imagesTr = join(out_dir, 'imagesTr')
    labelsTr = join(out_dir, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    labels, luts = build_tables()
    cases = []
    for subject in subdirs(join(source_dir, 'Imaging Data'), prefix='QUADRA_HC_', join=False):
        for session in ('Test', 'Retest'):
            if isfile(join(source_dir, 'Imaging Data', subject, session, f'{subject}_{session}_CT-AC.nii.gz')):
                cases.append(f'{subject}_{session}')

    with Pool(num_processes) as p:
        done = p.starmap(convert_case, [(source_dir, c, imagesTr, labelsTr, luts) for c in cases])
    assert len(done) == len(cases)
    write_dataset_json(out_dir, labels, len(cases))


def write_dataset_json(out_dir: str, labels: dict, num_cases: int):
    dataset_json = {
        'name': DATASET_NAME,
        'description': 'Total-body CT from [18F]FDG-PET/CT (Siemens Biograph Vision Quadra) of 48 healthy controls, each '
                       'scanned twice (test/retest, 96 cases), from the Medical University of Vienna QUADRA_HC dataset on '
                       'Zenodo. 86 labels (digestive tract, organs, lung lobes, muscles, ribs, vertebrae, hip bones, sacrum) '
                       'were generated automatically with MOOSE and only sanity-checked by the authors; this folder merges '
                       'five MOOSE segmentation files with fixed label IDs. Only the CT is included because the PET is not '
                       'on the CT grid.',
        'channel_names': {'0': 'CT'},
        'labels': labels,
        'file_ending': '.nii.gz',
        'numTraining': num_cases,
        'license': 'CC-BY-4.0',
        'commercial_ok': True,
        'reference': ['https://zenodo.org/records/16686025'],
        'citation': [
            'Medical University of Vienna (2025). Total-Body [18F]FDG-PET/CT Imaging of Healthy Controls: Test/Retest Data '
            'for Systemic, Multi-Organ Analysis (Version 1.1.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16686025'
        ],
        'converted_by': ['Fabian Isensee'],
        'release': '1.1.1 (Zenodo, 2025-08-02)',
        'note': "Reconverted on 2026-10-08 from Zenodo v1.1.1 (QUADRA_HC.zip, md5 5b37cd936988a023c67fe8f85d634d41) with "
                "nnU-Net master nnunetv2/dataset_conversion/Dataset239_QUADRA_HC.py. The earlier conversion (Moritz "
                "Langenberg) renumbered the label values per case, so labels from 83 on (and in a few cases from 9 or from "
                "the 12th ribs on) meant different structures in different cases, and it named the two kidneys the wrong way "
                "round. Mapping of the MOOSE files to label IDs: Digestive 1-4 -> 1-4, Organs 1-19 -> 5-23, Muscles 1-10 -> "
                "24-33, Ribs left 1-12 / right 1-12 / sternum -> 34-45 / 46-57 / 58 (the MOOSE 13th-rib classes are never "
                "present and are not included), Vertebrae 1-28 -> 59-86; where files overlap (median about 400, at most about 2,100 voxels "
                "per case) the later file in this order wins. Some classes are absent in some cases: vertebrae L6 (83) in 73 "
                "of 96 cases (where present mostly only a few voxels), gallbladder (9) in both scans of QUADRA_HC_010, left "
                "12th rib (45) in both scans of QUADRA_HC_037 and right 12th rib (57) in QUADRA_HC_037_Retest. The hip bone "
                "labels (84, 85) are inaccurate: they extend beyond the bone towards the femur (visual check Fabian Isensee, "
                "2026-10-08). From the authors: 'QUADRA_HC data consists of 48 healthy controls (absence of diseases + no "
                "history of diseases) who were scanned in a test/retest setting with a fixed protocol. The segmentations were "
                "automatically generated by moose and we performed only simple checks. For example, if the sides are correct "
                "and all volumes are there and structurally sound. No voxel-accurate corrections or checks were performed. "
                "The dataset is more geared towards subsequent analysis of healthy controls to get a better understanding of "
                "the healthy metabolism captured by FDG PET/CT and to assess the repeatability of such scans.' The release "
                "also contains PET (PT-SUV, not on the CT grid) and Body-Composition, Cardiac and Peripheral-Bones "
                "segmentations of low quality; these are not included. The random 5-fold split that nnU-Net generates cannot "
                "be used here: it puts the test and retest scan of the same subject into training and validation. Split "
                "manually by subject (QUADRA_HC_XXX).",
    }
    save_json(dataset_json, join(out_dir, 'dataset.json'), sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help="The extracted QUADRA_HC folder of the Zenodo release (contains 'Imaging Data')")
    parser.add_argument('-o', type=str, default=None,
                        help='Output base folder, the dataset is created as a subfolder. Default: nnUNet_raw')
    parser.add_argument('-np', type=int, default=12, help='Number of processes. Default: 12')
    args = parser.parse_args()
    convert(args.input_folder, args.o if args.o is not None else str(nnUNet_raw), args.np)
