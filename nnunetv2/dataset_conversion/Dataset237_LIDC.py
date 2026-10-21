#    Copyright 2026 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""
LIDC-IDRI (https://www.cancerimagingarchive.net/collection/lidc-idri/): lung nodule segmentation in thoracic CT, with
labels regenerated from the original reader annotations (XML). Supersedes Dataset022_LIDC, whose masks were made by an
undocumented procedure. The images are not copied: dataset.json references the images of Dataset022_LIDC via the
dataset key (one CT scan per patient, 1010 cases), except for two cases whose Dataset022 images are truncated (see
Data handling).

Inputs (from the TCIA collection page):
- LIDC-XML-only.zip, extracted into LIDC_FOLDER/LIDC-XML-only (contains tcia-lidc-xml/ and the corrected
  161-resubmitted-correction-3-9-12.xml). Optionally also lidc-idri-nodule-counts-6-23-2015.xlsx and
  tcia-diagnosis-data-2012-04-20.xls in LIDC_FOLDER; they are copied into the dataset folder.
- The LIDC-IDRI DICOM images (DICOM_FOLDER/LIDC-IDRI-XXXX/<study>/<series>/*.dcm). They map the image SOP UIDs
  referenced by the annotations to image slices; the pixel data of some slices (all slices of irregular series, see
  below) is compared with the Dataset022 images.
- Dataset022_LIDC (images are referenced, their geometry is used for the labels).
Run
    python Dataset237_LIDC.py -i LIDC_FOLDER -d DICOM_FOLDER
Requires pydicom (not an nnU-Net dependency).

How LIDC was annotated (Armato et al. 2011, Med Phys 38(2):915; McNitt-Gray et al. 2007, Acad Radiol 14(12):1464;
LIDC reader instructions and XML documentation on the TCIA page):
- Each scan was read by 4 thoracic radiologists (from 12 at 5 sites), first blinded, then unblinded: each reader saw the
  anonymized marks of the other three and finalized their own marks independently. There was no forced consensus. The
  XML files contain only the final (unblinded) marks, one readingSession per reader. Reader order is not consistent
  across scans.
- Lesion categories:
  'nodule >= 3 mm' (greatest in-plane diameter 3-30 mm): outlined on every slice, plus 9 subjective ratings
  (subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy;
  malignancy 1-5 rated assuming a 60-year-old male smoker).
  'nodule < 3 mm' (not clearly benign): a single approximate center point, no outline, no ratings.
  'non-nodule >= 3 mm' (any other pulmonary lesion, e.g. scars; also masses > 30 mm): a single point, no ratings.
- The outline is drawn on the first pixels outside the nodule. Outlines with inclusion=FALSE mark regions (e.g.
  cavities) that are excluded from the nodule; their outline pixels belong to the excluded region.
- Readers' nodule IDs are reader-specific. LIDC does not provide which marks of different readers are the same lesion.
- Patient-level diagnoses exist for 157 patients only (tcia-diagnosis-data-2012-04-20.xls) and cannot be linked to
  individual nodules.

Labels (labelsTr):
- Each reader's mask is the union of their 'nodule >= 3 mm' annotations, rasterized on the image grid: polygons
  filled, outline pixels removed, inclusion=FALSE regions (incl. their outline) removed. XML pixel coordinates are
  0-based column (x) / row (y); the slice is given by the referenced SOP UID (see Data handling).
- 1 (nodule): voxels in the masks of >= 2 readers. Dataset022 mostly followed this rule but counted the outline pixels
  as nodule; its masks are not exactly reproducible and are empty for the 14 scans with flipped in-plane orientation
  (README.md in the dataset folder lists the differences).
- 2 (ignore):
  - nodules outlined by only one reader, dilated by 3 voxels so that the whole lesion is covered;
  - small nodules ('nodule < 3 mm' marks): a 3 mm ball around the point.
  Label 1 is the ground truth including its border, so ignore never touches it: where one reader's mask extends beyond
  a label 1 nodule, or a small nodule mark lies on it, that stays background.
- 0 (background): everything else, including 'non-nodule' marks (their extent is unknown).
Additionally written:
- readersTr/CASE.nii.gz: uint8, bit r set = voxel in the mask of reader r (r = readingSession order in the XML), so
  other agreement rules can be derived without parsing the XML again.
- readersTr_single/CASE_R.nii.gz: the same masks as one binary file per reader, R = r + 1.
- nodules.json: per case the matched lesions (marks of different readers grouped into one lesion) with all reader
  marks, ratings, centroids and label voxel counts. Matching (only used for this file, not for the labels): 'nodule
  >= 3 mm' annotations of different readers are joined if their masks (incl. outline) overlap, greedily by IoU and never
  two annotations of the same reader in one lesion; points are attached to such a lesion if within 3 mm of it,
  remaining points of different readers are joined if within 5 mm.

Data handling:
- Chest radiograph reads (IdriReadMessage XML) are ignored. Duplicate XML files of the same series must have identical
  reading sessions and are read once. 161-resubmitted-correction-3-9-12.xml (TCIA correction 2012-03-09) replaces
  tcia-lidc-xml/157/161.xml (LIDC-IDRI-0101).
- Rois reference their image by SOP UID. Some reading sessions (LIDC-IDRI-0017, 0659) use SOP UIDs that are not in the
  series, but their z positions are exactly those of the series' slices; these rois are located by z position.
  Reading sessions with no roi in the series (neither by SOP UID nor by z position) are dropped: LIDC-IDRI-0566 has 8
  reading sessions, sessions 5-8 reference other images at z positions that do not exist in the series.
- LIDC-IDRI-0700 has only 3 readers (threshold stays >= 2).
- Contours with fewer than 3 points have no interior and contribute nothing. Point marks outside the image are dropped.
- Patients with two CT scans (0132, 0151, 0315, 0332, 0355, 0365, 0442, 0484): only the scan in Dataset022 is used.
- Mapping annotations to the Dataset022 images: the DICOM series is identified by its in-plane geometry. Regular series
  (one DICOM slice per image slice, on the image's z grid) are mapped by z position; first, middle and last slice are
  verified by pixel data. Some series are irregular (repeated acquisitions at the same z position, z shifts within the
  series, slices far outside the scanned range). Dataset022 contains a subset of their slices on a uniform z grid, so
  their DICOM slices are mapped to image slices by identical pixel data, and marks on DICOM slices that are not in the
  image are dropped (listed under 'issues' in nodules.json).
- The Dataset022 images of LIDC-IDRI-0085 and LIDC-IDRI-0672 are truncated: the converter that made them kept only
  one gap-free block of slices. 0085 consists of two acquisitions that share one z position (Dataset022: 137 of 268
  DICOM slices), 0672 lacks three slices (Dataset022: 273 of 462; a nodule outlined by all 4 readers is in the missing
  part). These two are converted from DICOM into imagesTr on the z grid of the Dataset022 image: at the repeated z
  position the slice contained in Dataset022 is used, the missing slices are the mean of their neighbors.
"""
import argparse
import hashlib
import os
import shutil
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from glob import glob
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import pydicom
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p, save_json, subfiles
from scipy import ndimage
from skimage.draw import polygon
from skimage.morphology import ball

from nnunetv2.paths import nnUNet_raw

NS = '{http://www.nih.gov}'
CORRECTION_XML = '161-resubmitted-correction-3-9-12.xml'
MIN_READERS = 2
POINT_IGNORE_RADIUS_MM = 3.
IGNORE_DILATION_VOXELS = 3
POINT_TO_NODULE_DISTANCE_MM = 3.
POINT_TO_POINT_DISTANCE_MM = 5.
EXTRA_FILES = ('lidc-idri-nodule-counts-6-23-2015.xlsx', 'tcia-diagnosis-data-2012-04-20.xls')
# Dataset022 images that hold only part of their DICOM series; these are converted from DICOM into imagesTr
TRUNCATED_IN_D22 = ('lidc_0085a', 'lidc_0672a')


def _text(element, tag):
    return element.find(NS + tag).text.strip()


def _parse_session(session):
    """Turns one readingSession into plain python objects (contour annotations, point annotations, non-nodules)."""
    contours, points, non_nodules = [], [], []
    for nodule in session.findall(NS + 'unblindedReadNodule'):
        rois = [{'sop_uid': _text(roi, 'imageSOP_UID'),
                 'z': float(_text(roi, 'imageZposition')),
                 'inclusion': _text(roi, 'inclusion').upper() == 'TRUE',
                 'xy': [(int(e.find(NS + 'xCoord').text), int(e.find(NS + 'yCoord').text))
                        for e in roi.findall(NS + 'edgeMap')]}
                for roi in nodule.findall(NS + 'roi')]
        characteristics = nodule.find(NS + 'characteristics')
        characteristics = None if characteristics is None or len(characteristics) == 0 else \
            {c.tag.replace(NS, ''): int(c.text) for c in characteristics}
        annotation = {'xml_id': _text(nodule, 'noduleID'), 'characteristics': characteristics, 'rois': rois}
        # 'nodule < 3 mm' marks are a single roi with a single point. Two marks in the data have ratings (all 0) but
        # are a single point as well; they are treated as points.
        if len(rois) == 1 and len(rois[0]['xy']) == 1:
            points.append(annotation)
        else:
            contours.append(annotation)
    for non_nodule in session.findall(NS + 'nonNodule'):
        locus = non_nodule.find(NS + 'locus')
        non_nodules.append({'xml_id': _text(non_nodule, 'nonNoduleID'), 'characteristics': None,
                            'rois': [{'sop_uid': _text(non_nodule, 'imageSOP_UID'),
                                      'z': float(_text(non_nodule, 'imageZposition')), 'inclusion': True,
                                      'xy': [(int(_text(locus, 'xCoord')), int(_text(locus, 'yCoord')))]}]})
    return {'contours': contours, 'points': points, 'non_nodules': non_nodules}


def load_ct_reads(xml_folder: str):
    """
    Returns {series instance uid: (xml file, [parsed reading session, ...])} for all CT series. Duplicate XML files must
    have identical reading sessions. The TCIA correction file replaces the file it corrects.
    """
    def signature(sessions):
        return [tuple((e.tag, (e.text or '').strip()) for e in s.iter()) for s in sessions]

    reads = {}
    for f in sorted(glob(join(xml_folder, 'tcia-lidc-xml', '**', '*.xml'), recursive=True)):
        root = ET.parse(f).getroot()
        if root.tag != NS + 'LidcReadMessage':  # IdriReadMessage: chest radiograph reads
            continue
        series_uid = _text(root.find(NS + 'ResponseHeader'), 'SeriesInstanceUid')
        sessions = root.findall(NS + 'readingSession')
        if series_uid in reads:
            assert signature(sessions) == reads[series_uid][2], \
                f'{f} and {reads[series_uid][0]} belong to the same series but differ'
            continue
        reads[series_uid] = (os.path.relpath(f, xml_folder), sessions, signature(sessions))
    root = ET.parse(join(xml_folder, CORRECTION_XML)).getroot()
    series_uid = _text(root.find(NS + 'ResponseHeader'), 'SeriesInstanceUid')
    assert series_uid in reads
    reads[series_uid] = (CORRECTION_XML, root.findall(NS + 'readingSession'), None)
    return {k: (f, [_parse_session(s) for s in sessions]) for k, (f, sessions, _) in reads.items()}


def read_ct_series(patient_folder: str):
    """{series instance uid: [dicom header, ...]} of all CT series of a patient (headers only)."""
    series = defaultdict(list)
    for series_folder in sorted(glob(join(patient_folder, '*', '*'))):
        for f in sorted(glob(join(series_folder, '*.dcm'))):
            h = pydicom.dcmread(f, stop_before_pixels=True)
            if h.Modality != 'CT':
                break
            series[h.SeriesInstanceUID].append(h)
    return series


def in_plane_match(headers, origin, spacing, size, direction):
    """Same orientation, matrix, pixel spacing and first-pixel position (x, y) as the image."""
    # row and column direction of the DICOM pixel grid = direction of image axes x and y
    image_orientation = np.array(direction).reshape(3, 3)[:, :2].T.ravel()
    return all(np.allclose(np.array(h.ImageOrientationPatient, float), image_orientation, atol=1e-4) and
               (h.Columns, h.Rows) == tuple(size[:2]) and
               np.allclose((float(h.PixelSpacing[1]), float(h.PixelSpacing[0])), spacing[:2], rtol=1e-4) and
               np.allclose(np.array(h.ImagePositionPatient[:2], float), origin[:2], atol=1e-2) for h in headers)


def _hounsfield(header):
    ds = pydicom.dcmread(header.filename)
    return np.round(ds.pixel_array * float(ds.get('RescaleSlope', 1)) +
                    float(ds.get('RescaleIntercept', 0))).astype(np.int32)


def map_slices(headers, image_file, origin, spacing, size, tolerance=0.05):
    """
    {sop instance uid: image slice index} and notes. A regular series (one DICOM slice per image slice, all on the
    image's z grid) is mapped by z position and its first, middle and last slice are checked against the image pixel
    data. Some series are irregular (repeated acquisitions at the same z position, z shifts within the series, slices
    outside the image); Dataset022 kept a subset of their slices on a uniform grid, so these are mapped by identical
    pixel data.
    """
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_file)).astype(np.int32)
    k_z = [(float(h.ImagePositionPatient[2]) - origin[2]) / spacing[2] for h in headers]
    if len(headers) == size[2] and len({round(k) for k in k_z}) == size[2] and \
            all(abs(k - round(k)) <= tolerance and 0 <= round(k) < size[2] for k in k_z):
        order = np.argsort(k_z)
        for i in (order[0], order[len(order) // 2], order[-1]):
            if not np.array_equal(_hounsfield(headers[i]), image[round(k_z[i])]):
                raise RuntimeError(f'{image_file}: image slice {round(k_z[i])} does not have the pixel data of the '
                                   f'DICOM slice at its z position')
        return {h.SOPInstanceUID: int(round(k)) for h, k in zip(headers, k_z)}, []

    slices_by_content = defaultdict(list)
    for k in range(size[2]):
        slices_by_content[hashlib.sha1(image[k].tobytes()).hexdigest()].append(k)
    sop_to_slice, max_shift = {}, 0.
    for h, kz in zip(headers, k_z):
        hits = slices_by_content.get(hashlib.sha1(_hounsfield(h).tobytes()).hexdigest())
        if hits:
            sop_to_slice[h.SOPInstanceUID] = k = min(hits, key=lambda k: abs(k - kz))
            max_shift = max(max_shift, abs(k - kz) * spacing[2])
    notes = [f'irregular DICOM series ({len(headers)} slices for {size[2]} image slices), slices mapped by pixel data: '
             f'{len(headers) - len(sop_to_slice)} DICOM slices are not in the image, '
             f'{size[2] - len(set(sop_to_slice.values()))} image slices match no DICOM slice, image slices are up to '
             f'{max_shift:.2f} mm away from the z position of their DICOM slice']
    return sop_to_slice, notes


def write_full_image(headers, d22_image_file, output_file):
    """
    Writes the whole DICOM series on the z grid of a truncated Dataset022 image (LIDC-IDRI-0085: two acquisitions
    sharing one z position, Dataset022 kept one of them; LIDC-IDRI-0672: three slices missing, Dataset022 kept the last
    gap-free block). At a repeated z position the slice contained in Dataset022 is used, missing slices are the mean
    of their two neighbors. Returns notes.
    """
    d22 = sitk.ReadImage(d22_image_file)
    d22_array = sitk.GetArrayFromImage(d22)
    d22_slices = {hashlib.sha1(s.astype(np.int32).tobytes()).hexdigest() for s in d22_array}
    z = np.array([float(h.ImagePositionPatient[2]) for h in headers])
    k = (z - z.min()) / d22.GetSpacing()[2]
    assert np.all(np.abs(k - np.round(k)) < 0.05), f'{d22_image_file}: DICOM slices are not on a common z grid'
    k = np.round(k).astype(int)
    array = np.zeros((k.max() + 1,) + d22_array.shape[1:], np.int32)
    filled = np.zeros(len(array), bool)
    for i in np.argsort(k):
        hu = _hounsfield(headers[i])
        if not filled[k[i]] or hashlib.sha1(hu.tobytes()).hexdigest() in d22_slices:
            array[k[i]] = hu
            filled[k[i]] = True
    missing = np.nonzero(~filled)[0]
    assert all(0 < m < len(array) - 1 and filled[m - 1] and filled[m + 1] for m in missing)
    for m in missing:
        array[m] = np.round((array[m - 1] + array[m + 1]) / 2)
    img = sitk.GetImageFromArray(array.astype(d22_array.dtype))
    img.SetSpacing(d22.GetSpacing())
    img.SetDirection(d22.GetDirection())
    img.SetOrigin(d22.GetOrigin()[:2] + (float(z.min()),))
    sitk.WriteImage(img, output_file, True)
    return [f'the Dataset022 image holds {len(d22_array)} of the {len(array)} slice positions of the DICOM series; the '
            f'whole series was converted into imagesTr' +
            (f' ({len(missing)} missing slices filled with the mean of their neighbors)' if len(missing) else '')]


def rasterize_contour_annotation(annotation, shape_zyx):
    """
    Flat voxel indices of one 'nodule >= 3 mm' annotation (rois with their image slice index): (mask, mask including
    inclusion outlines). The mask follows the LIDC reader instructions: the outline is the first pixels outside the
    nodule, inclusion=FALSE outlines remove their interior and outline.
    """
    plane = shape_zyx[1:]
    by_slice = defaultdict(list)
    for roi in annotation['rois']:
        by_slice[roi['slice']].append(roi)
    mask, mask_with_outline = [], []
    for k, rois in by_slice.items():
        inside, with_outline, excluded = (np.zeros(plane, bool) for _ in range(3))
        for roi in rois:
            xy = np.array(roi['xy'])
            outline = np.zeros(plane, bool)
            outline[np.clip(xy[:, 1], 0, plane[0] - 1), np.clip(xy[:, 0], 0, plane[1] - 1)] = True
            filled = np.zeros(plane, bool)
            if len(xy) >= 3:
                filled[polygon(xy[:, 1], xy[:, 0], plane)] = True
            if roi['inclusion']:
                inside |= filled & ~outline
                with_outline |= filled | outline
            else:
                excluded |= filled | outline
        for target, m in ((mask, inside & ~excluded), (mask_with_outline, with_outline)):
            yy, xx = np.nonzero(m)
            target.append(np.ravel_multi_index((np.full_like(yy, k), yy, xx), shape_zyx))
    return np.unique(np.concatenate(mask)), np.unique(np.concatenate(mask_with_outline))


def dilate(mask, structure):
    """binary_dilation restricted to the bounding boxes of the connected components (masks are sparse)."""
    r = structure.shape[0] // 2
    labeled, _ = ndimage.label(mask)
    out = np.zeros_like(mask)
    for bbox in ndimage.find_objects(labeled):
        bbox = tuple(slice(max(s.start - r, 0), min(s.stop + r, dim)) for s, dim in zip(bbox, mask.shape))
        out[bbox] |= ndimage.binary_dilation(mask[bbox], structure)
    return out


def _greedy_merge(n_items, readers, candidate_pairs):
    """Union-find over candidate pairs (score, i, j) in the given order; never puts two items of one reader together."""
    parent = list(range(n_items))
    members = {i: {readers[i]} for i in range(n_items)}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for _, i, j in candidate_pairs:
        ri, rj = find(i), find(j)
        if ri != rj and not members[ri] & members[rj]:
            parent[rj] = ri
            members[ri] |= members.pop(rj)
    groups = defaultdict(list)
    for i in range(n_items):
        groups[find(i)].append(i)
    return list(groups.values())


def match_lesions(contours, points, spacing_zyx, shape_zyx):
    """
    Groups the marks of different readers into lesions. contours: [(reader, mask with outline (flat indices))],
    points: [(reader, (z, y, x))]. Returns [(contour indices, point indices)].
    """
    pairs = []
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            if contours[i][0] != contours[j][0]:
                overlap = np.intersect1d(contours[i][1], contours[j][1], assume_unique=True).size
                if overlap:
                    pairs.append((overlap / (contours[i][1].size + contours[j][1].size - overlap), i, j))
    groups = _greedy_merge(len(contours), [c[0] for c in contours], sorted(pairs, reverse=True))
    lesions = [(g, []) for g in groups]

    spacing_zyx = np.array(spacing_zyx)
    lesion_voxels_mm = [np.concatenate([np.array(np.unravel_index(contours[i][1], shape_zyx)).T for i in g]) *
                        spacing_zyx for g in groups]
    unattached = []
    for p, (_, zyx) in enumerate(points):
        distances = [np.min(np.linalg.norm(v - np.array(zyx) * spacing_zyx, axis=1)) for v in lesion_voxels_mm]
        if distances and min(distances) <= POINT_TO_NODULE_DISTANCE_MM:
            lesions[int(np.argmin(distances))][1].append(p)
        else:
            unattached.append(p)
    pairs = []
    for a in range(len(unattached)):
        for b in range(a + 1, len(unattached)):
            i, j = unattached[a], unattached[b]
            d = np.linalg.norm((np.array(points[i][1]) - np.array(points[j][1])) * spacing_zyx)
            if points[i][0] != points[j][0] and d <= POINT_TO_POINT_DISTANCE_MM:
                pairs.append((-d, a, b))
    for g in _greedy_merge(len(unattached), [points[p][0] for p in unattached], sorted(pairs, reverse=True)):
        lesions.append(([], [unattached[a] for a in g]))
    return lesions


def _point_ball_offsets(spacing_zyx, radius_mm):
    r = [int(np.ceil(radius_mm / s)) for s in spacing_zyx]
    grid = np.stack(np.meshgrid(*[np.arange(-ri, ri + 1) for ri in r], indexing='ij'), -1).reshape(-1, 3)
    return grid[np.linalg.norm(grid * np.array(spacing_zyx), axis=1) <= radius_mm]


def make_labels(readers, num_readers, small_nodules_zyx, spacing_zyx):
    """
    readers: uint8, bit r = voxel in the mask of reader r. small_nodules_zyx: voxel positions of 'nodule < 3 mm' marks.
    1 = voxels in the masks of >= 2 readers: the ground truth, including its border.
    2 = ignore: nodules outlined by only one reader, dilated by 3 voxels to cover the whole lesion, and a 3 mm ball
        around each small nodule mark. Single-reader voxels and small nodule marks touching label 1 are part of that
        nodule and stay background; ignore keeps at least one voxel away from label 1.
    0 = everything else.
    """
    count = sum(((readers >> r) & 1).astype(np.uint8) for r in range(num_readers))
    seg = (count >= MIN_READERS).astype(np.uint8)
    near_label_1 = dilate(seg == 1, np.ones((3, 3, 3), bool))
    single = count == 1
    components, _ = ndimage.label(single | (seg == 1), np.ones((3, 3, 3)))
    single &= ~np.isin(components, np.unique(components[seg == 1]))
    ignore = dilate(single, ball(IGNORE_DILATION_VOXELS).astype(bool))
    offsets = _point_ball_offsets(spacing_zyx, POINT_IGNORE_RADIUS_MM)
    for zyx in small_nodules_zyx:
        idx = np.array(zyx) + offsets
        idx = tuple(idx[np.all((idx >= 0) & (idx < np.array(seg.shape)), axis=1)].T)
        if not near_label_1[idx].any():
            ignore[idx] = True
    seg[ignore & ~near_label_1] = 2
    return seg


def reader_physical_point(reader, index_xyz):
    """World coordinate (mm) of a continuous voxel index, from the header in an sitk.ImageFileReader."""
    return np.array(reader.GetOrigin()) + np.array(reader.GetDirection()).reshape(3, 3) @ \
        (np.array(index_xyz) * np.array(reader.GetSpacing()))


def _header(image_file):
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_file)
    reader.ReadImageInformation()
    return reader


def _write(arr, reference_reader, filename):
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(reference_reader.GetOrigin())
    img.SetSpacing(reference_reader.GetSpacing())
    img.SetDirection(reference_reader.GetDirection())
    sitk.WriteImage(img, filename, True)


_READS = None


def _set_reads(reads):
    global _READS
    _READS = reads


def convert_case(case, image_file, patient_folder, output_folder, reads=None):
    """reads: output of load_ct_reads; worker processes get it once via _set_reads."""
    reads = _READS if reads is None else reads
    reader = _header(image_file)
    origin, spacing, size, direction = reader.GetOrigin(), reader.GetSpacing(), reader.GetSize(), reader.GetDirection()
    # axial images with ascending z; some series have row/column directions (-1, 0, 0) / (0, -1, 0)
    assert np.allclose(np.abs(direction), np.eye(3).ravel()) and direction[8] > 0, f'{case}: unexpected direction'

    series = read_ct_series(patient_folder)
    matches = [uid for uid, headers in series.items() if in_plane_match(headers, origin, spacing, size, direction)]
    assert len(matches) == 1, f'{case}: {len(matches)} DICOM series match the image geometry'
    series_uid = matches[0]
    issues = []
    if case in TRUNCATED_IN_D22:
        new_image_file = join(output_folder, 'imagesTr', case + '_0000.nii.gz')
        issues += write_full_image(series[series_uid], image_file, new_image_file)
        image_file = new_image_file
        reader = _header(image_file)
        origin, size = reader.GetOrigin(), reader.GetSize()
    shape_zyx, spacing_zyx = size[::-1], spacing[::-1]
    sop_to_slice, notes = map_slices(series[series_uid], image_file, origin, spacing, size)
    issues += notes
    xml_file, sessions = reads[series_uid]

    # Rois reference their image by SOP UID. Some reading sessions (e.g. LIDC-IDRI-0017, 0659) use SOP UIDs that are not
    # in the series although their z positions are exactly those of its slices; these are located by z position.
    series_z = np.array([float(h.ImagePositionPatient[2]) for h in series[series_uid]])
    series_sops = [h.SOPInstanceUID for h in series[series_uid]]
    series_sop_set = set(series_sops)

    def locate(roi):
        """('uid' | 'z' | None, image slice index or None if that DICOM slice is not in the image)"""
        if roi['sop_uid'] in series_sop_set:
            return 'uid', sop_to_slice.get(roi['sop_uid'])
        same_z = np.nonzero(np.abs(series_z - roi['z']) < 0.01)[0]
        if len(same_z) == 0:
            return None, None
        slices = {sop_to_slice[series_sops[i]] for i in same_z if series_sops[i] in sop_to_slice}
        return 'z', slices.pop() if len(slices) == 1 else None

    # drop reading sessions that belong to another series and marks on DICOM slices that are not in the image
    used_sessions, dropped_sessions = [], []
    for s, session in enumerate(sessions):
        located = {kind: [[(roi, *locate(roi)) for roi in a['rois']] for a in session[kind]]
                   for kind in ('contours', 'points', 'non_nodules')}
        found_by = Counter(how for kind in located.values() for a in kind for _, how, _ in a)
        if sum(found_by.values()) and not found_by['uid'] + found_by['z']:
            dropped_sessions.append(s)
            continue
        if found_by['z']:
            issues.append(f'reading session {s}: {found_by["z"]} roi(s) reference SOP UIDs that are not in the series, '
                          f'located by z position')
        kept = {}
        for kind in ('contours', 'points', 'non_nodules'):
            kept[kind] = []
            for a, rois in zip(session[kind], located[kind]):
                not_in_series = sum(how is None for _, how, _ in rois)
                not_in_image = sum(how is not None and k is None for _, how, k in rois)
                if not_in_series:
                    issues.append(f'reading session {s}: {not_in_series} roi(s) of {kind} {a["xml_id"]} are not in the '
                                  f'series, ignored')
                if not_in_image:
                    issues.append(f'reading session {s}: {not_in_image} roi(s) of {kind} {a["xml_id"]} are on DICOM '
                                  f'slices that are not in the image, ignored')
                rois = [{**roi, 'slice': k} for roi, _, k in rois if k is not None]
                if rois:
                    kept[kind].append({**a, 'rois': rois})
        used_sessions.append((s, kept))
    if dropped_sessions:
        issues.append(f'reading sessions {dropped_sessions} reference no image of this series, dropped')

    # per-reader masks (bit r of `readers`) from the 'nodule >= 3 mm' annotations
    readers = np.zeros(shape_zyx, np.uint8)
    flat = readers.reshape(-1)
    contour_marks, point_marks = [], []
    for r, (s, session) in enumerate(used_sessions):
        for a in session['contours']:
            mask, mask_with_outline = rasterize_contour_annotation(a, shape_zyx)
            flat[mask] |= np.uint8(1 << r)
            slices = sorted({roi['slice'] for roi in a['rois']})
            contour_marks.append((r, mask_with_outline, {
                'reader': r, 'type': 'nodule >= 3 mm', 'xml_id': a['xml_id'], 'characteristics': a['characteristics'],
                'num_voxels': int(mask.size), 'num_slices': len(slices)}))
            if mask.size == 0:
                issues.append(f'reading session {s}: nodule {a["xml_id"]} has no voxels after removing the outline')
        for kind, label in (('points', 'nodule < 3 mm'), ('non_nodules', 'non-nodule')):
            for a in session[kind]:
                roi = a['rois'][0]
                x, y = roi['xy'][0]
                zyx = (roi['slice'], y, x)
                if not all(0 <= c < d for c, d in zip(zyx, shape_zyx)):
                    issues.append(f'reading session {s}: {label} {a["xml_id"]} at x={x} y={y} is outside the image, '
                                  f'dropped')
                    continue
                entry = {'reader': r, 'type': label, 'xml_id': a['xml_id'], 'voxel_zyx': list(map(int, zyx))}
                if a['characteristics'] is not None:
                    entry['characteristics'] = a['characteristics']
                point_marks.append((r, zyx, entry))
    seg = make_labels(readers, len(used_sessions),
                      [zyx for _, zyx, entry in point_marks if entry['type'] == 'nodule < 3 mm'], spacing_zyx)
    _write(seg, reader, join(output_folder, 'labelsTr', case + '.nii.gz'))
    _write(readers, reader, join(output_folder, 'readersTr', case + '.nii.gz'))
    for r in range(len(used_sessions)):
        _write((readers >> r) & 1, reader, join(output_folder, 'readersTr_single', f'{case}_{r + 1}.nii.gz'))

    # lesion list for nodules.json
    seg_flat = seg.reshape(-1)
    lesions = []
    for contour_idx, point_idx in match_lesions([(r, m) for r, m, _ in contour_marks],
                                                [(r, zyx) for r, zyx, _ in point_marks], spacing_zyx, shape_zyx):
        marks = [contour_marks[i][2] for i in contour_idx] + [point_marks[i][2] for i in point_idx]
        n = Counter(m['type'] for m in marks)
        category = 'nodule >= 3 mm' if n['nodule >= 3 mm'] else ('nodule < 3 mm' if n['nodule < 3 mm'] else
                                                                  'non-nodule')
        if contour_idx:
            voxels = np.unique(np.concatenate([contour_marks[i][1] for i in contour_idx]))
            zyx = np.array(np.unravel_index(voxels, shape_zyx))
        else:
            voxels = None
            zyx = np.array([point_marks[i][1] for i in point_idx]).T
        centroid = zyx.mean(1)
        lesion = {'category': category,
                  'num_readers': {k: n[k] for k in ('nodule >= 3 mm', 'nodule < 3 mm', 'non-nodule')},
                  'centroid_voxel_zyx': [round(float(c), 2) for c in centroid],
                  'centroid_world_xyz': [round(float(c), 2) for c in reader_physical_point(reader, centroid[::-1])]}
        if voxels is not None:
            lesion['bbox_voxel_zyx'] = [[int(c.min()), int(c.max()) + 1] for c in zyx]
            lesion['num_voxels_label_1'] = int((seg_flat[voxels] == 1).sum())
        lesion['marks'] = marks
        lesions.append(lesion)
    order = {'nodule >= 3 mm': 0, 'nodule < 3 mm': 1, 'non-nodule': 2}
    lesions.sort(key=lambda l: (order[l['category']], -l['num_readers'][l['category']], l['centroid_voxel_zyx']))
    lesions = [{'lesion_id': i + 1, **l} for i, l in enumerate(lesions)]
    return case, {
        'patient_id': os.path.basename(patient_folder),
        'series_instance_uid': series_uid,
        'xml_file': xml_file,
        'num_readers': len(used_sessions),
        'issues': issues,
        'num_voxels': {'nodule': int((seg == 1).sum()), 'ignore': int((seg == 2).sum())},
        'lesions': lesions,
    }


def _convert_case_or_error(case, *args, attempts=3):
    """
    Retries (reading from network shares occasionally fails, with truncated reads or EAGAIN) and lets all cases run
    before failing; returns (case, traceback) on error.
    """
    for _ in range(attempts):
        try:
            return convert_case(case, *args)
        except Exception:
            error = traceback.format_exc()
            time.sleep(30)
    return case, error


README = """# Dataset237_LIDC

LIDC-IDRI lung nodule segmentation in thoracic CT, labels regenerated from the original reader XML annotations
(TCIA LIDC-XML-only.zip). Supersedes Dataset022_LIDC. Images are referenced from Dataset022_LIDC via the dataset key
of dataset.json (1010 cases, one CT scan per patient), except LIDC-IDRI-0085 and 0672 (imagesTr, see below). Converted
with nnunetv2/dataset_conversion/Dataset237_LIDC.py; its docstring describes the annotation process and every
conversion decision in detail.

## Labels (labelsTr)
- Every reader's mask = union of their 'nodule >= 3 mm' outlines, filled, outline pixels removed (LIDC reader
  instructions: the outline is the first pixels outside the nodule), inclusion=FALSE regions removed.
- 1 nodule: voxels in the masks of >= 2 readers. This is the ground truth, including its border.
- 2 ignore: nodules outlined by only one reader (dilated by 3 voxels), and small nodules ('nodule < 3 mm' marks: 3 mm
  ball around the point). Ignore never touches label 1.
- 0 background: everything else, including 'non-nodule' marks (scars, masses > 30 mm, ...).

## Other files
- readersTr/: uint8 per case, bit r = voxel in the mask of reader r (XML reading session order; reader identity is
  not consistent across scans).
- readersTr_single/: the same reader masks as binary files CASE_R.nii.gz, R = r + 1 (1-4; LIDC-IDRI-0700 has 3).
- nodules.json: per case the matched lesions with all reader marks (type, XML nodule id, ratings incl. malignancy,
  voxel position), centroid, bounding box and label voxel counts, plus conversion issues.
- lidc-idri-nodule-counts-6-23-2015.xlsx: TCIA per-scan lesion counts. tcia-diagnosis-data-2012-04-20.xls:
  patient-level diagnosis (157 patients; cannot be linked to individual nodules).

## Differences to Dataset022_LIDC
Dataset022's masks mostly are the voxels inside the outlines of >= 2 readers with the outline pixels counted as nodule
(so they are larger), but not exactly: they contain some single-reader nodules, miss some nodules outlined by >= 2
readers, and are empty for all 14 scans with flipped in-plane orientation (LIDC-IDRI-0218, 0262, 0291, 0311, 0313,
0415, 0422, 0423, 0671, 0697, 0785, 0796, 0850, 0938; 12 of them contain nodules outlined by >= 2 readers) and for
LIDC-IDRI-0606 and 0765.

## Known issues of the (Dataset022) images
- Some DICOM series are irregular (repeated acquisitions at the same z position, z shifts, stray slices); Dataset022
  holds a subset of their slices on a uniform z grid. Annotations were mapped by pixel data; marks on slices that are
  not in the image were dropped (see 'issues' in nodules.json).
- The Dataset022 images of LIDC-IDRI-0085 (137 of 268 DICOM slices; two acquisitions sharing one z position) and
  LIDC-IDRI-0672 (273 of 462; three slices missing in the series) are truncated: only one gap-free block of slices was
  kept. These two cases were converted from DICOM into imagesTr of this dataset (repeated z position: the slice in
  Dataset022 is used; missing slices: mean of their neighbors); dataset.json points to them.
"""


def convert_lidc(lidc_folder: str, dicom_folder: str, output_folder: str, d22_folder: str, num_processes: int = 8):
    reads = load_ct_reads(join(lidc_folder, 'LIDC-XML-only'))
    maybe_mkdir_p(join(output_folder, 'labelsTr'))
    maybe_mkdir_p(join(output_folder, 'readersTr'))
    maybe_mkdir_p(join(output_folder, 'readersTr_single'))
    maybe_mkdir_p(join(output_folder, 'imagesTr'))

    cases = sorted(i[:-len('_0000.nii.gz')] for i in subfiles(join(d22_folder, 'imagesTr'), suffix='_0000.nii.gz',
                                                              join=False))
    jobs = [(c, join(d22_folder, 'imagesTr', c + '_0000.nii.gz'),
             join(dicom_folder, f'LIDC-IDRI-{c[len("lidc_"):len("lidc_") + 4]}'), output_folder) for c in cases]
    with Pool(num_processes, initializer=_set_reads, initargs=(reads,)) as p:
        outcomes = p.starmap(_convert_case_or_error, jobs)
    failed = {case: error for case, error in outcomes if isinstance(error, str)}
    if failed:
        raise RuntimeError('\n\n'.join(f'{case}:\n{error}' for case, error in failed.items()))
    results = dict(outcomes)
    unused_series = sorted(set(reads) - {r['series_instance_uid'] for r in results.values()})

    save_json(results, join(output_folder, 'nodules.json'), sort_keys=False)
    for f in EXTRA_FILES:
        if isfile(join(lidc_folder, f)):
            shutil.copy(join(lidc_folder, f), output_folder)
    write_dataset_json_and_readme(output_folder, d22_folder, cases)
    return results, unused_series


def write_dataset_json_and_readme(output_folder: str, d22_folder: str, cases):
    with open(join(output_folder, 'README.md'), 'w') as f:
        f.write(README)
    d22_name = os.path.basename(os.path.normpath(d22_folder))
    dataset_name = os.path.basename(os.path.normpath(output_folder))
    dataset_json = {
        'name': dataset_name,
        'description': 'Thoracic CT scans from the LIDC-IDRI collection (TCIA), one scan per patient (1010 cases), '
                       'with lung nodule labels regenerated from the original reader annotations (TCIA '
                       'LIDC-XML-only.zip): nodule = voxels inside the \'nodule >= 3 mm\' outlines of at least 2 of '
                       'the 4 readers, ignore = nodules outlined by only one reader and small (< 3 mm) nodules. This '
                       'dataset '
                       f'supersedes {d22_name}, whose masks were made by an undocumented procedure. Images are '
                       f'referenced from {d22_name} via the dataset key; only the labels are stored here, except for '
                       'LIDC-IDRI-0085 and 0672, whose images are truncated in ' + d22_name + ' and were converted '
                       'from DICOM into imagesTr.',
        'channel_names': {'0': 'CT'},
        'labels': {'background': 0, 'nodule': 1, 'ignore': 2},
        'file_ending': '.nii.gz',
        'numTraining': len(cases),
        'license': 'CC-BY-3.0',
        'commercial_ok': True,
        'reference': [
            'https://www.cancerimagingarchive.net/collection/lidc-idri/',
            'https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX',
        ],
        'citation': [
            'Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, et al. (2015). Data From '
            'LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX',
            'Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, et al. (2011). The Lung Image '
            'Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database '
            'of Lung Nodules on CT Scans. Medical Physics 38(2), 915-931. https://doi.org/10.1118/1.3528204',
        ],
        'converted_by': ['Fabian Isensee'],
        'release': 'TCIA LIDC-IDRI version 4 (2020/09/21), annotations from LIDC-XML-only.zip',
        'note': 'Supersedes ' + d22_name + '. Converted with nnunetv2/dataset_conversion/Dataset237_LIDC.py (nnU-Net '
                'master branch); its docstring and README.md in this folder document the LIDC annotation process and '
                'how the labels were made. Each scan was read by 4 radiologists (blinded, then unblinded, no forced '
                'consensus) who outlined every \'nodule >= 3 mm\' (3-30 mm) and marked \'nodule < 3 mm\' and '
                '\'non-nodule\' lesions with a single point. Labels: per reader the outlines are filled with the '
                'outline pixels removed (the outline is the first pixels outside the nodule) and inclusion=FALSE '
                'regions removed. 1 = voxels of >= 2 readers (the ground truth, including its border). 2 (ignore) = '
                'nodules outlined by only one reader (dilated by 3 voxels) and small nodules (3 mm ball around each '
                '\'nodule < 3 mm\' mark); ignore never touches label 1. 0 = everything else, including \'non-nodule\' '
                'marks (incl. masses > 30 mm). Compared with these labels, the masks of ' + d22_name +
                ' mostly are the voxels inside the outlines of >= 2 readers with the outline pixels counted as nodule, '
                'but not exactly: they contain some single-reader nodules, miss some nodules outlined by >= 2 readers, '
                'and are empty for all 14 scans with flipped in-plane orientation (0218, 0262, 0291, 0311, 0313, 0415, '
                '0422, 0423, 0671, 0697, 0785, 0796, 0850, 0938; 12 of them contain nodules outlined by >= 2 readers) '
                'and for 0606 and 0765. readersTr/ holds the per-reader masks as bits (bit r = reader r), '
                'readersTr_single/ the same as one binary file per reader (CASE_R, R = r + 1), nodules.json the '
                'lesions matched across readers with '
                'all reader marks and ratings (incl. subjective malignancy 1-5). Patients with two CT scans (0132, '
                '0151, 0315, 0332, 0355, 0365, 0442, 0484): only the scan in ' + d22_name + ' is used. Some DICOM '
                'series are irregular (repeated acquisitions, z shifts, stray slices) and ' + d22_name + ' holds a '
                'subset of their slices; annotations were mapped by pixel data and marks on missing slices dropped. '
                'The ' + d22_name + ' images of LIDC-IDRI-0085 (137 of 268 DICOM slices; the series consists of two '
                'acquisitions sharing one z position) and LIDC-IDRI-0672 (273 of 462; three slices are missing in the '
                'series; a nodule outlined by all 4 readers is in the missing part) are truncated, so these two '
                'cases were converted from DICOM into imagesTr (repeated z position: the slice in ' + d22_name +
                ' is used; missing slices: mean of their neighbors). TCIA asks '
                'for the acknowledgement: "The authors acknowledge the National Cancer Institute and the Foundation '
                'for the National Institutes of Health, and their critical role in the creation of the free publicly '
                'available LIDC/IDRI Database used in this study."',
        'dataset': {c: {'images': [f'./imagesTr/{c}_0000.nii.gz' if c in TRUNCATED_IN_D22 else
                                   f'$nnUNet_raw/{d22_name}/imagesTr/{c}_0000.nii.gz'],
                        'label': f'./labelsTr/{c}.nii.gz'} for c in cases},
    }
    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--lidc_folder', type=str, required=True,
                        help='Folder with the extracted LIDC-XML-only.zip (LIDC-XML-only/) and optionally the TCIA '
                             'spreadsheets')
    parser.add_argument('-d', '--dicom_folder', type=str, required=True,
                        help='LIDC-IDRI DICOM folder (contains LIDC-IDRI-XXXX/)')
    parser.add_argument('-o', '--output_folder', type=str, default=join(nnUNet_raw, 'Dataset237_LIDC'),
                        help='default: $nnUNet_raw/Dataset237_LIDC')
    parser.add_argument('--d22', type=str, default=join(nnUNet_raw, 'Dataset022_LIDC'),
                        help='Dataset022_LIDC, whose images are referenced. default: $nnUNet_raw/Dataset022_LIDC')
    parser.add_argument('-np', type=int, default=8, help='number of processes, default: 8')
    args = parser.parse_args()
    results, unused_series = convert_lidc(args.lidc_folder, args.dicom_folder, args.output_folder, args.d22, args.np)
    print(f'{len(results)} cases; {len(unused_series)} annotated CT series not used (second scans): {unused_series}')
    for case, r in results.items():
        for issue in r['issues']:
            print(f'{case}: {issue}')
