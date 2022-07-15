#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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


from multiprocessing import Pool
from multiprocessing.dummy import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from skimage.io import imread
from skimage.io import imsave
from skimage.morphology import ball
from skimage.morphology import erosion
from skimage.transform import resize

from nnunet.paths import nnUNet_raw_data
from nnunet.paths import preprocessing_output_dir


def load_bmp_convert_to_nifti_borders(img_file, lab_file, img_out_base, anno_out, spacing, border_thickness=0.7):
    img = imread(img_file)
    img_itk = sitk.GetImageFromArray(img.astype(np.float32))
    img_itk.SetSpacing(np.array(spacing)[::-1])
    sitk.WriteImage(img_itk, join(img_out_base + "_0000.nii.gz"))

    if lab_file is not None:
        l = imread(lab_file)
        borders = generate_border_as_suggested_by_twollmann(l, spacing, border_thickness)
        l[l > 0] = 1
        l[borders == 1] = 2
        l_itk = sitk.GetImageFromArray(l.astype(np.uint8))
        l_itk.SetSpacing(np.array(spacing)[::-1])
        sitk.WriteImage(l_itk, anno_out)


def generate_ball(spacing, radius, dtype=int):
    radius_in_voxels = np.round(radius / np.array(spacing)).astype(int)
    n = 2 * radius_in_voxels + 1
    ball_iso = ball(max(n) * 2, dtype=np.float64)
    ball_resampled = resize(ball_iso, n, 1, 'constant', 0, clip=True, anti_aliasing=False, preserve_range=True)
    ball_resampled[ball_resampled > 0.5] = 1
    ball_resampled[ball_resampled <= 0.5] = 0
    return ball_resampled.astype(dtype)


def generate_border_as_suggested_by_twollmann(label_img: np.ndarray, spacing, border_thickness: float = 2) -> np.ndarray:
    border = np.zeros_like(label_img)
    selem = generate_ball(spacing, border_thickness)
    for l in np.unique(label_img):
        if l == 0: continue
        mask = (label_img == l).astype(int)
        eroded = erosion(mask, selem)
        border[(eroded == 0) & (mask != 0)] = 1
    return border


def find_differences(labelstr1, labelstr2):
    for n in subfiles(labelstr1, suffix='.nii.gz', join=False):
        a = sitk.GetArrayFromImage(sitk.ReadImage(join(labelstr1, n)))
        b = sitk.GetArrayFromImage(sitk.ReadImage(join(labelstr2, n)))
        print(n, np.sum(a != b))


def plot_images(folder, output_folder):
    maybe_mkdir_p(output_folder)
    import matplotlib.pyplot as plt
    for i in subfiles(folder, suffix='.nii.gz', join=False):
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(folder, i)))
        center_slice = img[img.shape[0]//2]
        plt.imsave(join(output_folder, i[:-7] + '.png'), center_slice)


def convert_to_tiff(nifti_image: str, output_name: str):
    npy = sitk.GetArrayFromImage(sitk.ReadImage(nifti_image))
    imsave(output_name, npy.astype(np.uint16),  compress=6)


def convert_to_instance_seg(arr: np.ndarray, spacing: tuple = (0.2, 0.125, 0.125)):
    from skimage.morphology import label, dilation
    # 1 is core, 2 is border
    objects = label((arr == 1).astype(int))
    final = np.copy(objects)
    remaining_border = arr == 2
    current = np.copy(objects)
    dilated_mm = np.array((0, 0, 0))
    spacing = np.array(spacing)

    while np.sum(remaining_border) > 0:
        strel_size = [0, 0, 0]
        maximum_dilation = max(dilated_mm)
        for i in range(3):
            if spacing[i] == min(spacing):
                strel_size[i] = 1
                continue
            if dilated_mm[i] + spacing[i] / 2 < maximum_dilation:
                strel_size[i] = 1
        ball_here = ball(1)

        if strel_size[0] == 0: ball_here = ball_here[1:2]
        if strel_size[1] == 0: ball_here = ball_here[:, 1:2]
        if strel_size[2] == 0: ball_here = ball_here[:, :, 1:2]

        #print(1)
        dilated = dilation(current, ball_here)
        diff = (current == 0) & (dilated != current)
        final[diff & remaining_border] = dilated[diff & remaining_border]
        remaining_border[diff] = 0
        current = dilated
        dilated_mm = [dilated_mm[i] + spacing[i] if strel_size[i] == 1 else dilated_mm[i] for i in range(3)]
    return final.astype(np.uint32)


def convert_to_instance_seg2(arr: np.ndarray, spacing: tuple = (0.2, 0.125, 0.125), small_center_threshold=30,
                             isolated_border_as_separate_instance_threshold: int = 15):
    from skimage.morphology import label, dilation
    # we first identify centers that are too small and set them to be border. This should remove false positive instances
    objects = label((arr == 1).astype(int))
    for o in np.unique(objects):
        if o > 0 and np.sum(objects == o) <= small_center_threshold:
            arr[objects == o] = 2

    # 1 is core, 2 is border
    objects = label((arr == 1).astype(int))
    final = np.copy(objects)
    remaining_border = arr == 2
    current = np.copy(objects)
    dilated_mm = np.array((0, 0, 0))
    spacing = np.array(spacing)

    while np.sum(remaining_border) > 0:
        strel_size = [0, 0, 0]
        maximum_dilation = max(dilated_mm)
        for i in range(3):
            if spacing[i] == min(spacing):
                strel_size[i] = 1
                continue
            if dilated_mm[i] + spacing[i] / 2 < maximum_dilation:
                strel_size[i] = 1
        ball_here = ball(1)

        if strel_size[0] == 0: ball_here = ball_here[1:2]
        if strel_size[1] == 0: ball_here = ball_here[:, 1:2]
        if strel_size[2] == 0: ball_here = ball_here[:, :, 1:2]

        #print(1)
        dilated = dilation(current, ball_here)
        diff = (current == 0) & (dilated != current)
        final[diff & remaining_border] = dilated[diff & remaining_border]
        remaining_border[diff] = 0
        current = dilated
        dilated_mm = [dilated_mm[i] + spacing[i] if strel_size[i] == 1 else dilated_mm[i] for i in range(3)]

    # what can happen is that a cell is so small that the network only predicted border and no core. This cell will be
    # fused with the nearest other instance, which we don't want. Therefore we identify isolated border predictions and
    # give them a separate instance id
    # we identify isolated border predictions by checking each foreground object in arr and see whether this object
    # also contains label 1
    max_label = np.max(final)

    foreground_objects = label((arr != 0).astype(int))
    for i in np.unique(foreground_objects):
        if i > 0 and (1 not in np.unique(arr[foreground_objects==i])):
            size_of_object = np.sum(foreground_objects==i)
            if size_of_object >= isolated_border_as_separate_instance_threshold:
                final[foreground_objects == i] = max_label + 1
                max_label += 1
                #print('yeah boi')

    return final.astype(np.uint32)


def load_instanceseg_save(in_file: str, out_file:str, better: bool):
    itk_img = sitk.ReadImage(in_file)
    if not better:
        instanceseg = convert_to_instance_seg(sitk.GetArrayFromImage(itk_img))
    else:
        instanceseg = convert_to_instance_seg2(sitk.GetArrayFromImage(itk_img))
    itk_out = sitk.GetImageFromArray(instanceseg)
    itk_out.CopyInformation(itk_img)
    sitk.WriteImage(itk_out, out_file)


def convert_all_to_instance(input_folder: str, output_folder: str, processes: int = 24, better: bool = False):
    maybe_mkdir_p(output_folder)
    p = Pool(processes)
    files = subfiles(input_folder, suffix='.nii.gz', join=False)
    output_files = [join(output_folder, i) for i in files]
    input_files = [join(input_folder, i) for i in files]
    better = [better] * len(files)
    r = p.starmap_async(load_instanceseg_save, zip(input_files, output_files, better))
    _ = r.get()
    p.close()
    p.join()


if __name__ == "__main__":
    source_train = '/home/isensee/drives/E132-Rohdaten/CellTrackingChallenge/train/Fluo-N3DH-SIM+'
    source_test = '/home/isensee/drives/E132-Rohdaten/CellTrackingChallenge/test/Fluo-N3DH-SIM+'

    task_id = 76
    task_name = 'Fluo_N3DH_SIM'
    spacing = (0.2, 0.125, 0.125)
    border_thickness = 0.5
    p = Pool(12)

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    res = []

    for train_sequence in ['01', '02']:
        train_cases = subfiles(join(source_train, train_sequence), suffix=".tif", join=False)
        for t in train_cases:
            casename = train_sequence + "_" + t[:-4]
            img_file = join(source_train, train_sequence, t)
            lab_file = join(source_train, train_sequence + "_GT", "SEG", "man_seg" + t[1:])
            img_out_base = join(imagestr, casename)
            anno_out = join(labelstr, casename + ".nii.gz")
            res.append(
                p.starmap_async(load_bmp_convert_to_nifti_borders, ((img_file, lab_file, img_out_base, anno_out, spacing, border_thickness),)))
            train_patient_names.append(casename)

    for test_sequence in ['01', '02']:
        test_cases = subfiles(join(source_test, test_sequence), suffix=".tif", join=False)
        for t in test_cases:
            casename = test_sequence + "_" + t[:-4]
            img_file = join(source_test, test_sequence, t)
            lab_file = None
            img_out_base = join(imagests, casename)
            anno_out = None
            res.append(
                p.starmap_async(load_bmp_convert_to_nifti_borders, ((img_file, lab_file, img_out_base, anno_out, spacing, border_thickness),)))
            test_patient_names.append(casename)

    _ = [i.get() for i in res]

    json_dict = {}
    json_dict['name'] = task_name
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "BF",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "cell",
        "2": "border",
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    p.close()
    p.join()

    # We dont use this, ignore
    task_name = "Task076_Fluo_N3DH_SIM"
    labelsTr = join(nnUNet_raw_data, task_name, "labelsTr")
    cases = subfiles(labelsTr, suffix='.nii.gz', join=False)
    splits = []
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('01_')],
         'val': [i[:-7] for i in cases if i.startswith('02_')]}
    )
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('02_')],
         'val': [i[:-7] for i in cases if i.startswith('01_')]}
    )

    maybe_mkdir_p(join(preprocessing_output_dir, task_name))

    save_pickle(splits, join(preprocessing_output_dir, task_name, "splits_final.pkl"))

    # test set was converted to instance seg with convert_all_to_instance with better=True

    # convert to tiff with convert_to_tiff



