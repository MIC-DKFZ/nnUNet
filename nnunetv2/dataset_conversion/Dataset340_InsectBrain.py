"""
InsectBrain: micro-CT of ant heads with semi-manual brain masks (Toulkeridou et al. 2023), Dryad CC0.

Download (Dryad needs a browser or an API token; the API without token returns 401):
    https://datadryad.org/dataset/doi:10.5061/dryad.qz612jmgv (version 5, published 2023-10-03)
    README.md (sha-256 23ab03d4368cf3a47ce80af55928de66bc4da282f282dce27c6fe0dac6c1eea7)
    training.zip (9,323,214,894 bytes, sha-256 b01c95091581b99bee6fae92f2e80dd402c9ae0d1a50238eb63e2cd749d24425)
    testing.zip (4,617,236,781 bytes, sha-256 359900b0d67c71cf1c381b26074eaeda7ff94579933f6d4126208bc8dced8011)
    The browser download wraps each zip in another zip (doi_10_5061_dryad_qz612jmgv__v20231003.zip); unpack that first.
Extract training.zip and testing.zip into one folder (it then contains training/ and testing/; skip __MACOSX and ._*):
    python Dataset340_InsectBrain.py /path/to/extracted /path/to/README.md [-o output_base] [-np 8]
This writes Dataset340_InsectBrain (all stacks at Dryad resolution) and Dataset341_InsectBrain_resampled (340 at half
resolution).

What the release contains (checked 2026-10-09):
- One head scan per species. Each scan is published as stacks of 2D TIFF slices, one stack per scan direction:
  training/<name>_<i>.tif + <name>_<i>_mask.tif, where <name>, <name>2, <name>3 are the directions of the same scan.
  testing/ uses three naming schemes (<name>_<i>[_mask].tif, <name><i>.tif + <name>__mask<i>.tif, and raw Amira exports
  <name><i>.tif + <name>_mask<i>.tif with .info files).
- Almost all stacks are crops that were resized to 520x520 per slice WITHOUT keeping the aspect ratio (authors' notebook
  github.com/evropi/U-Net mask_generator_example.ipynb: iaa.Resize(520), linear/cubic, masks resized the same way, so
  mask values are 0-255). Each direction was cropped with its own box (the stacks of one scan do not overlay under any
  axis assignment, with or without shift), so the in-plane voxel size cannot be recovered. Along the stack axis the
  slices are at scan resolution. We therefore use spacing 1x1x1 for every case.
- Six testing stacks are raw Amira exports (uint16, about 1013x980 per slice, isotropic voxels, mask 2 = brain,
  3 = background): Acanthomyrmex, Eciton, Gnamptogenys, Leptogenys, Leptomyrmex, Ooceraea. Their shape is correct.
- 14 resized testing stacks have inverted masks (brain = 0, rest = 255), see INVERTED_MASKS; they are flipped here
  (Pseudomyrmex_ferrugineus only from slice 40 on, its slices 1-39 are not inverted).
- Many training stacks lack exactly one slice inside the index range (typically at index 51, 101, 151, ...). The slices
  next to the gap correlate like slices two apart, so a slice is really missing; we fill it by linear interpolation of
  the two neighbours (image and continuous mask, then threshold) and list the filled indices in the csv.
- Three stacks have one empty mask slice inside the brain (annotation missing, see UNANNOTATED_MASK_SLICES); these
  mask slices are interpolated from the neighbours like missing slices.
- Copies and broken stacks (see EXCLUDED and TRUNCATED; found by hashing all slices, correlating all stack pairs of a
  species and scoring how well mask edges lie on image edges):
  testing/Veromessor_lariversi3 is a copy of training/Veromessor_lariversi3 slices 172-250 (excluded);
  training/Trachymyrmex_carinatus2 is a copy of training/Trachymyrmex_carinatus slices 101-199 (excluded);
  training/Crematogaster_lineolata2 slices 281-400 are copies of training/Crematogaster_lineolata3 slices 281-400 (cut);
  training/Parasyscia_crib_rinodis2: the masks do not belong to the images (no slice shift, flip or transpose fits, and
  no other stack has them; excluded).
  testing/Dolichoderus_pustulatus3 holds other images than training/Dolichoderus_pustulatus3; it becomes stack 4.
- training/Dorymyrmex_insanus_s holds three stacks under one name (index ranges 1-400, 2100-2300, 3100-3400).
- Masks of the resized stacks are thresholded at > 127.

All stacks go to imagesTr. The official Dryad train/test assignment and the provenance of every case are in
official_train_test_split.csv. Several cases come from the same scan (one scan per species), so do NOT use nnU-Net's
random 5-fold split: splits_final.json (folds grouped by species) is written to the raw folder; copy it to
nnUNet_preprocessed/<dataset>/ before training.
"""
import argparse
import csv
import re
import shutil
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
import tifffile
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, subfiles

from nnunetv2.paths import nnUNet_raw

DATASET_NAME = 'Dataset340_InsectBrain'
DATASET_NAME_HALF = 'Dataset341_InsectBrain_resampled'

# training/: Dryad base name -> (species, specimen). Stacks are <base> (1), <base>2 (2), <base>3 (3).
# Species names follow the Dryad README table, typos in file names corrected.
TRAINING = {
    'Acromyrmex_versicolor': 'Acromyrmex_versicolor',
    'Atta_texana': 'Atta_texana',
    'Brachymyrmex_depilis': 'Brachymyrmex_depilis',
    'Camponotus_cf.neatricus': 'Camponotus_nearcticus',
    'Camponotus_modoc': 'Camponotus_modoc',
    'Carebara_affinis': 'Carebara_affinis',
    'Cephalotes_atratus': 'Cephalotes_atratus',
    'Cephalotes_minutus': 'Cephalotes_minutus',
    'Crematogaster_lineolata': 'Crematogaster_lineolata',
    'Crematogaster_pimicola': 'Crematogaster_pinicola',
    'Cyphomyrmex_flavidus': 'Cyphomyrmex_flavidus',
    'Daceton_armigerum': 'Daceton_armigerum',
    'Dolichoderus_mariae': 'Dolichoderus_mariae',
    'Dolichoderus_pustulatus': 'Dolichoderus_pustulatus',
    'Dorylus_kohli': 'Dorylus_kohli',
    'Dorymyrmex_insanus_group-l': ('Dorymyrmex_insanus', 'Dorymyrmex_insanus_large'),
    'Formica-polyctena': 'Formica_polyctena',
    'Formica_exsectoides': 'Formica_exsectoides',
    'Formica_gnava': 'Formica_gnava',
    'Formica_neogagates': 'Formica_neogagates',
    'Formica_rufa': 'Formica_rufa',
    'Labidus_praedator': 'Labidus_praedator',
    'Liometopum-apiculatum': 'Liometopum_apiculatum',
    'Monomorium_pharaonis': 'Monomorium_pharaonis',
    'Parasyscia_crib_rinodis': 'Parasyscia_cribrinodis',
    'Patagonomymex_angustus': 'Patagonomyrmex_angustus',
    'Pheidole_cf_morrisi': 'Pheidole_cf_morrisi',
    'Pheidole_rhea': 'Pheidole_rhea',
    'Pogonomyrmex_badius': 'Pogonomyrmex_badius',
    'Pogonomyrmex_brevispinosus': 'Pogonomyrmex_brevispinosus',
    'Pogonomyrmex_desertorum': 'Pogonomyrmex_desertorum',
    'Pogonomyrmex_magnacanthus': 'Pogonomyrmex_magnacanthus',
    'Pogonomyrmex_pima': 'Pogonomyrmex_pima',
    'Pogonomyrmex_rugosus': 'Pogonomyrmex_rugosus',
    'Pogonomyrmex_schmitti': 'Pogonomyrmex_schmitti',
    'Pseudomyrmex_ejectus': 'Pseudomyrmex_ejectus',
    'Pseudomyrmex_triplarinus': 'Pseudomyrmex_triplarinus',
    'Strumigenys_arizonicus': 'Strumigenys_arizonicus',
    'Temnothorax_curvispinosus': 'Temnothorax_curvispinosus',
    'Tetramorium_immigrans': 'Tetramorium_immigrans',
    'Trachymyrmex_carinatus': 'Trachymyrmex_carinatus',
    'Veromessor_andrei': 'Veromessor_andrei',
    'Veromessor_lariversi': 'Veromessor_lariversi',
    'Veromessor_lobognathus': 'Veromessor_lobognathus',
    'Veromessor_smithi': 'Veromessor_smithi',
}
# training/Dorymyrmex_insanus_s: three stacks in one index space
DORYMYRMEX_SMALL = ('Dorymyrmex_insanus_s', [(1, 400), (2100, 2300), (3100, 3400)])

# testing/: Dryad name (image prefix) -> (species, stack). PSMVG is assigned to Pseudomyrmex veneficus because its 370
# slices match that README row (inferred, not stated by Dryad).
TESTING = {
    'Acanthomyrmex-glabfemoralis': ('Acanthomyrmex_glabfemoralis', 1),
    'Aenictus_paradentatus': ('Aenictus_paradentatus', 1),
    'Anolgr': ('Anoplolepis_gracilipes', 1),
    'Camponotus-hyatti': ('Camponotus_hyatti', 1),
    'Camponotus-vicinus': ('Camponotus_vicinus', 1),
    'Carebara_atoma': ('Carebara_atoma', 1),
    'Cephalotes_sp': ('Cephalotes_sp', 1),
    'Dolichoderus_pustulatus3': ('Dolichoderus_pustulatus', 4),
    'Eci-brucheli': ('Eciton_burchellii', 1),
    'Formica-obscuripes': ('Formica_obscuripes', 1),
    'GesHow': ('Gesomyrmex_howardi', 1),
    'Gnamp_sp': ('Gnamptogenys_sp', 1),
    'Las_fuhi': ('Lasius_fuliginosus', 1),
    'Lept_peuq': ('Leptogenys_peuqueti', 1),
    'Leptomyrmex_darlingtoni': ('Leptomyrmex_darlingtoni', 1),
    'Linepithema_humile': ('Linepithema_humile', 1),
    'Myrmelachista_nodigera': ('Myrmelachista_nodigera', 1),
    'Mystrium_camillae': ('Mystrium_camillae', 1),
    'Oocera-biroi': ('Ooceraea_biroi', 1),
    'Orectognathus_versicolor': ('Orectognathus_versicolor', 1),
    'PSMVG': ('Pseudomyrmex_veneficus', 1),
    'Pheidbicoir': ('Pheidole_bicarinata', 1),
    'PristProf': ('Pristomyrmex_profundus', 1),
    'Pseudomyrmex_ferrugineus': ('Pseudomyrmex_ferrugineus', 1),
}
EXCLUDED = {
    ('testing', 'Veromessor_lariversi3'): 'copy of training/Veromessor_lariversi3 slices 172-250',
    ('training', 'Trachymyrmex_carinatus2'): 'copy of training/Trachymyrmex_carinatus slices 101-199',
    ('training', 'Parasyscia_crib_rinodis2'): 'masks do not belong to the images',
}
# (split, name) -> kept Dryad index range
TRUNCATED = {('training', 'Crematogaster_lineolata2'): (1, 280)}  # 281-400 are copies of Crematogaster_lineolata3
# (split, name) -> Dryad indices whose mask is empty although the brain is visible (neighbours 13-40 % brain)
UNANNOTATED_MASK_SLICES = {('testing', 'Aenictus_paradentatus'): [151], ('training', 'Pogonomyrmex_desertorum3'): [151],
                           ('training', 'Pheidole_rhea3'): [51]}
# raw Amira exports: mask prefix differs from the image prefix for Gnamp_sp
RAW_AMIRA = {'Acanthomyrmex-glabfemoralis', 'Eci-brucheli', 'Gnamp_sp', 'Lept_peuq', 'Leptomyrmex_darlingtoni',
             'Oocera-biroi'}
MASK_PREFIX = {'Gnamp_sp': 'Gamp_sp'}
# testing/ only (training/Dolichoderus_pustulatus3 is not inverted): brain = 0, everything else = 255 (checked visually
# on the middle slice of every testing stack and by the per-slice brain fraction). Value: inverted Dryad index range,
# None = whole stack.
INVERTED_MASKS = {'Anolgr': None, 'Camponotus-hyatti': None, 'Camponotus-vicinus': None, 'Carebara_atoma': None,
                  'Cephalotes_sp': None, 'Dolichoderus_pustulatus3': None, 'Formica-obscuripes': None, 'GesHow': None,
                  'Linepithema_humile': None, 'Mystrium_camillae': None, 'PSMVG': None, 'Pheidbicoir': None,
                  'PristProf': None, 'Pseudomyrmex_ferrugineus': (40, 400)}

TRAIN_RE = re.compile(r'^(?P<name>.+?)_(?P<idx>-?\d+)(?P<mask>_mask)?\.tif$')
TEST_RE = re.compile(r'^(?P<name>.+?)(?P<mask>_?_mask)?_?(?P<idx>\d+)(?P<mask2>_mask)?\.tif$')


def read_readme_voxel_sizes(readme: str) -> dict:
    sizes = {}
    for name, size in re.findall(r'^\| ([A-Z][\w.\\-]+) \|[^|\n]*\|[^|\n]*\|[^|\n]*\| *([\d.]+) *\|', open(readme).read(),
                                 re.M):
        sizes[name.replace('\\', '').replace('.', '_')] = float(size)
    # README spellings that differ from the species names used here
    sizes['Parasyscia_cribrinodis'] = sizes.pop('Parasyscia_cribrinobis')
    return sizes


def collect_stacks(source_dir: str) -> list:
    """Returns one dict per output case with the slice files (index -> (image, mask)) and provenance."""
    files = {}  # (split, name) -> {idx: {'img': path, 'mask': path}}
    for f in subfiles(join(source_dir, 'training'), suffix='.tif', join=False):
        m = TRAIN_RE.match(f)
        assert m, f
        files.setdefault(('training', m['name']), {}).setdefault(int(m['idx']), {})[
            'mask' if m['mask'] else 'img'] = join(source_dir, 'training', f)
    mask_prefix_to_name = {v: k for k, v in MASK_PREFIX.items()}
    for f in subfiles(join(source_dir, 'testing'), suffix='.tif', join=False):
        m = TEST_RE.match(f)
        assert m, f
        is_mask = bool(m['mask'] or m['mask2'])
        name = m['name']
        if is_mask:
            name = mask_prefix_to_name.get(name, name)
        files.setdefault(('testing', name), {}).setdefault(int(m['idx']), {})[
            'mask' if is_mask else 'img'] = join(source_dir, 'testing', f)

    stacks = []
    for (split, name), slices in sorted(files.items()):
        if (split, name) in EXCLUDED:
            continue
        if (split, name) in TRUNCATED:
            lo, hi = TRUNCATED[(split, name)]
            slices = {k: v for k, v in slices.items() if lo <= k <= hi}
        for k in UNANNOTATED_MASK_SLICES.get((split, name), []):
            slices[k] = {'img': slices[k]['img']}
        img_idx = [k for k, v in slices.items() if 'img' in v]
        mask_idx = [k for k, v in slices.items() if 'mask' in v]
        assert (min(img_idx), max(img_idx)) == (min(mask_idx), max(mask_idx)), f'{split}/{name}: index ranges differ'
        if split == 'testing':
            species, stack = TESTING[name]
            parts = [(species, species, stack, slices)]
        elif name == DORYMYRMEX_SMALL[0]:
            parts = [('Dorymyrmex_insanus', 'Dorymyrmex_insanus_small', i + 1,
                      {k: v for k, v in slices.items() if lo <= k <= hi})
                     for i, (lo, hi) in enumerate(DORYMYRMEX_SMALL[1])]
            assert sum(len(p[3]) for p in parts) == len(slices)
        else:
            if name in TRAINING:
                base, stack = name, 1
            else:
                base, stack = name[:-1], int(name[-1])
                assert base in TRAINING and stack in (2, 3), name
            species = TRAINING[base]
            species, specimen = species if isinstance(species, tuple) else (species, species)
            parts = [(species, specimen, stack, slices)]
        for species, specimen, stack, sl in parts:
            stacks.append({'case': f'{specimen}_{stack}', 'species': species, 'specimen': specimen, 'stack': stack,
                           'split': split, 'dryad_name': name, 'slices': sl,
                           'raw': split == 'testing' and name in RAW_AMIRA,
                           'inverted': split == 'testing' and name in INVERTED_MASKS,
                           'inverted_range': INVERTED_MASKS.get(name) if split == 'testing' else None})
    cases = [s['case'] for s in stacks]
    assert len(cases) == len(set(cases))
    assert {s['dryad_name'] for s in stacks if s['split'] == 'testing'} == set(TESTING)
    return stacks


def _fill_gaps(slices: dict, interpolate) -> list:
    """Fills missing indices inside the index range in place by linear interpolation; returns the filled indices."""
    idx = sorted(slices)
    filled = [i for i in range(idx[0], idx[-1] + 1) if i not in slices]
    for i in filled:
        lo = max(j for j in idx if j < i)
        hi = min(j for j in idx if j > i)
        slices[i] = interpolate(slices[lo], slices[hi], (i - lo) / (hi - lo))
    return filled


def load_stack(st: dict):
    imgs = {i: tifffile.imread(v['img']) for i, v in st['slices'].items() if 'img' in v}
    masks = {i: tifffile.imread(v['mask']) for i, v in st['slices'].items() if 'mask' in v}
    if st['raw']:
        for i, m in masks.items():
            assert set(np.unique(m)) <= {2, 3}, (st['case'], i)
        masks = {i: (m == 2).astype(np.float32) * 255 for i, m in masks.items()}
    else:
        lo, hi = st['inverted_range'] or (-np.inf, np.inf)
        masks = {i: (255 - m if st['inverted'] and lo <= i <= hi else m).astype(np.float32) for i, m in masks.items()}
    # image and mask can miss different slices (training/Atta_texana2: image 150 and mask 151 are missing)
    filled_img = _fill_gaps(imgs, lambda a, b, w: np.round((1 - w) * a.astype(np.float32) + w * b).astype(a.dtype))
    filled_mask = _fill_gaps(masks, lambda a, b, w: (1 - w) * a + w * b)
    assert sorted(imgs) == sorted(masks), st['case']
    order = sorted(imgs)
    img = np.stack([imgs[i] for i in order])
    seg = (np.stack([masks[i] for i in order]) > 127).astype(np.uint8)
    assert img.shape == seg.shape, st['case']
    if filled_img == filled_mask:
        filled = ' '.join(map(str, filled_img))
    else:
        filled = ' '.join([f'{i} (image)' for i in filled_img] + [f'{i} (mask)' for i in filled_mask])
    return img, seg, filled


def write_nifti(arr: np.ndarray, filename: str):
    itk = sitk.GetImageFromArray(arr)
    itk.SetSpacing((1., 1., 1.))
    sitk.WriteImage(itk, filename, True)


def downsample_2x(img: np.ndarray, seg: np.ndarray):
    """2x2x2 block mean for the image, 2x2x2 majority (ties -> brain) for the mask; odd trailing voxels dropped."""
    z, y, x = (s // 2 * 2 for s in img.shape)
    blocks = lambda a: a[:z, :y, :x].reshape(z // 2, 2, y // 2, 2, x // 2, 2)
    img_h = np.round(blocks(img).mean(axis=(1, 3, 5), dtype=np.float64)).astype(img.dtype)
    seg_h = (blocks(seg).sum(axis=(1, 3, 5)) >= 4).astype(np.uint8)
    return img_h, seg_h


def convert_stack(st: dict, out_dir: str, out_dir_half: str):
    img, seg, filled = load_stack(st)
    write_nifti(img, join(out_dir, 'imagesTr', f"{st['case']}_0000.nii.gz"))
    write_nifti(seg, join(out_dir, 'labelsTr', f"{st['case']}.nii.gz"))
    img_h, seg_h = downsample_2x(img, seg)
    write_nifti(img_h, join(out_dir_half, 'imagesTr', f"{st['case']}_0000.nii.gz"))
    write_nifti(seg_h, join(out_dir_half, 'labelsTr', f"{st['case']}.nii.gz"))
    return st['case'], filled, img.shape, img_h.shape, str(img.dtype), float(seg.mean())


def make_splits(stacks: list, n_folds: int = 5) -> list:
    """Folds grouped by species, greedily balanced by number of cases (largest species first)."""
    by_species = {}
    for s in stacks:
        by_species.setdefault(s['species'], []).append(s['case'])
    folds = [[] for _ in range(n_folds)]
    for species in sorted(by_species, key=lambda k: (-len(by_species[k]), k)):
        min(folds, key=len).extend(sorted(by_species[species]))
    all_cases = sorted(c for f in folds for c in f)
    return [{'train': [c for c in all_cases if c not in set(f)], 'val': sorted(f)} for f in folds]


def write_csv(stacks: list, results: dict, voxel_sizes: dict, filename: str, shape_key: int):
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'species', 'specimen', 'dryad_split', 'dryad_files', 'dryad_slice_indices',
                    'filled_missing_slices', 'source_format', 'mask_note', 'scan_voxel_size_um_readme', 'shape_zyx',
                    'brain_fraction'])
        for s in sorted(stacks, key=lambda s: s['case']):
            r = results[s['case']]
            idx = sorted(s['slices'])
            if s['raw']:
                fmt, note = 'raw Amira export (uint16, isotropic voxels)', 'Dryad mask value 2 = brain, 3 = background'
            else:
                fmt = 'crop resized to 520x520 per slice (uint8, in-plane voxel size unknown)'
                note = ''
                if s['inverted']:
                    rng = s['inverted_range']
                    note = 'inverted in Dryad (brain = 0), flipped' + (f' (slices {rng[0]}-{rng[1]})' if rng else '')
                note = '; '.join(filter(None, [note, 'Dryad mask 0-255 (interpolated), thresholded > 127']))
            key = (s['split'], s['dryad_name'])
            if key in UNANNOTATED_MASK_SLICES:
                note += f"; empty mask (not annotated) at {' '.join(map(str, UNANNOTATED_MASK_SLICES[key]))}, interpolated"
            if key in TRUNCATED:
                note += f'; only slices {TRUNCATED[key][0]}-{TRUNCATED[key][1]} used (later slices copy another stack)'
            pattern = f"{s['split']}/{s['dryad_name']}"
            w.writerow([s['case'], s['species'], s['specimen'], s['split'], pattern, f'{idx[0]}..{idx[-1]}',
                        r[1], fmt, note, voxel_sizes.get(s['specimen'], voxel_sizes.get(s['species'], '')),
                        'x'.join(map(str, r[shape_key])), f'{r[5]:.4f}'])


def dataset_json(name: str, num_cases: int, num_species: int, half: bool) -> dict:
    description = (
        f'Micro-CT scans of ant heads ({num_species} species, one head scan per species) with semi-manual binary segmentation of '
        'the brain, from the Dryad dataset of Toulkeridou et al. 2023. Dryad publishes each scan as stacks of 2D slices '
        'along up to three directions; every stack is one case here (training and testing data of the release, all in '
        'imagesTr). Most stacks are crops resized to 520x520 per slice, so voxels are not isotropic and the in-plane '
        'voxel size is unknown.')
    if half:
        description += f' Half-resolution copy of {DATASET_NAME} (2x2x2 block mean / majority vote).'
    note = (
        'Converted on 2026-10-09 from Dryad doi:10.5061/dryad.qz612jmgv version 5 (training.zip, testing.zip, sha-256 '
        'verified) with nnU-Net master nnunetv2/dataset_conversion/Dataset340_InsectBrain.py; replaces Dataset131_InsectBrain / '
        'Dataset132_InsectBrain_resampled (subset of the training data only, conversion not documented). One case per '
        'Dryad slice stack, named <specimen>_<stack>; cases of the same species are the same scan seen from different '
        'directions. Do not use the random 5-fold split: splits_final.json in this folder groups the folds by species '
        '(copy it to the nnUNet_preprocessed folder). official_train_test_split.csv lists for every case the Dryad '
        'train/test assignment, source files, filled slices, mask handling and the scan voxel size from the Dryad README '
        '(the official split is not species-disjoint: Dolichoderus pustulatus has stacks in both). Spacing is 1x1x1 for '
        'all cases: along the stack axis a voxel is one scan voxel, in-plane the 520x520 crops were resized per '
        'direction with unknown crop sizes (not recoverable from the release); only the six raw Amira testing stacks '
        '(Acanthomyrmex, Eciton, Gnamptogenys, Leptogenys, Leptomyrmex, Ooceraea) have their true isotropic geometry. '
        'Fixes applied: 14 testing stacks with inverted masks flipped (Pseudomyrmex ferrugineus from slice 40 on); resized '
        'masks (0-255) thresholded at > 127; three mask slices left empty in Dryad interpolated; single '
        'missing slices inside a stack filled by linear interpolation; Dryad stacks that copy other stacks excluded '
        '(testing/Veromessor_lariversi3, training/Trachymyrmex_carinatus2) or cut (training/Crematogaster_lineolata2 '
        'after slice 280); training/Parasyscia_crib_rinodis2 excluded (its masks do not belong to its images); '
        'training/Dorymyrmex_insanus_s split into its three stacks (small worker; '
        'Dorymyrmex_insanus_large is a second specimen of the same species). Images keep the Dryad intensities (uint8 '
        'for the resized stacks, uint16 for the raw stacks). Species names follow the Dryad README; Pheidole cf. morrisi, '
        'Strumigenys arizonicus, Veromessor lobognathus and Cephalotes sp. are not in the README table, PSMVG was assigned '
        'to Pseudomyrmex veneficus by its slice count. '
        'Optional citation (Dryad data citation): Toulkeridou E, Gutierrez CE, Baum D, Doya K, Economo EP (2023). '
        'Automated segmentation of insect anatomy from micro-CT images using deep learning [Dataset]. Dryad. '
        'https://doi.org/10.5061/dryad.qz612jmgv')
    if half:
        note = (f'Half resolution of {DATASET_NAME}: each axis halved (odd trailing voxel dropped), image = mean of '
                f'each 2x2x2 block (rounded, original dtype), mask = brain if at least 4 of the 8 voxels are brain. '
                f'Spacing kept at 1x1x1. ') + note
    return {
        'name': name,
        'description': description,
        'channel_names': {'0': 'microCT'},
        'labels': {'background': 0, 'ant brain': 1},
        'file_ending': '.nii.gz',
        'numTraining': num_cases,
        'license': 'CC0-1.0',
        'commercial_ok': True,
        'reference': ['https://datadryad.org/dataset/doi:10.5061/dryad.qz612jmgv'],
        'citation': [
            'Toulkeridou E, Gutierrez CE, Baum D, Doya K, Economo EP (2023). Automated segmentation of insect anatomy '
            'from micro-CT images using deep learning. Natural Sciences 3, e20230010. https://doi.org/10.1002/ntls.20230010'
        ],
        'converted_by': ['Fabian Isensee'],
        'release': 'Dryad version 5 (published 2023-10-03)',
        'note': note,
    }


def convert(source_dir: str, readme: str, output_base: str, num_processes: int = 8):
    out_dir = join(output_base, DATASET_NAME)
    out_dir_half = join(output_base, DATASET_NAME_HALF)
    for d in (out_dir, out_dir_half):
        maybe_mkdir_p(join(d, 'imagesTr'))
        maybe_mkdir_p(join(d, 'labelsTr'))

    voxel_sizes = read_readme_voxel_sizes(readme)
    stacks = collect_stacks(source_dir)
    with Pool(num_processes) as p:
        results = {r[0]: r for r in p.starmap(convert_stack, [(s, out_dir, out_dir_half) for s in stacks])}
    assert len(results) == len(stacks)

    splits = make_splits(stacks)
    for d, name, half in ((out_dir, DATASET_NAME, False), (out_dir_half, DATASET_NAME_HALF, True)):
        save_json(dataset_json(name, len(stacks), len({s['species'] for s in stacks}), half), join(d, 'dataset.json'),
                  sort_keys=False)
        save_json(splits, join(d, 'splits_final.json'), sort_keys=False)
        write_csv(stacks, results, voxel_sizes, join(d, 'official_train_test_split.csv'), 3 if half else 2)
        shutil.copy(readme, join(d, 'dryad_README.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='folder with the extracted training/ and testing/ folders')
    parser.add_argument('readme', type=str, help='README.md of the Dryad release (voxel sizes)')
    parser.add_argument('-o', type=str, default=nnUNet_raw, help='output base folder (default: nnUNet_raw)')
    parser.add_argument('-np', type=int, default=8, help='number of processes')
    args = parser.parse_args()
    convert(args.input_folder, args.readme, args.o, args.np)
