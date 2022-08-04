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

import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from skimage.io import imread
from skimage.io import imsave
from skimage.morphology import disk
from skimage.morphology import erosion
from skimage.transform import resize

from nnunet.paths import nnUNet_raw_data

from argparse import ArgumentParser


def load_bmp_convert_to_nifti_borders_2d(img_file, lab_file, img_out_base, anno_out, spacing, border_thickness=0.7):
    img = imread(img_file)
    img_itk = sitk.GetImageFromArray(img.astype(np.float32)[None])
    img_itk.SetSpacing(list(spacing)[::-1] + [999])
    sitk.WriteImage(img_itk, join(img_out_base + "_0000.nii.gz"))

    if lab_file is not None:
        l = imread(lab_file)
        borders = generate_border_as_suggested_by_twollmann_2d(l, spacing, border_thickness)
        l[l > 0] = 1
        l[borders == 1] = 2
        l_itk = sitk.GetImageFromArray(l.astype(np.uint8)[None])
        l_itk.SetSpacing(list(spacing)[::-1] + [999])
        sitk.WriteImage(l_itk, anno_out)


def generate_disk(spacing, radius, dtype=int):
    radius_in_voxels = np.round(radius / np.array(spacing)).astype(int)
    n = 2 * radius_in_voxels + 1
    disk_iso = disk(max(n) * 2, dtype=np.float64)
    disk_resampled = resize(disk_iso, n, 1, 'constant', 0, clip=True, anti_aliasing=False, preserve_range=True)
    disk_resampled[disk_resampled > 0.5] = 1
    disk_resampled[disk_resampled <= 0.5] = 0
    return disk_resampled.astype(dtype)


def generate_border_as_suggested_by_twollmann_2d(label_img: np.ndarray, spacing,
                                                 border_thickness: float = 2) -> np.ndarray:
    border = np.zeros_like(label_img)
    selem = generate_disk(spacing, border_thickness)
    for l in np.unique(label_img):
        if l == 0: continue
        mask = (label_img == l).astype(int)
        eroded = erosion(mask, selem)
        border[(eroded == 0) & (mask != 0)] = 1
    return border


def convert_to_instance_seg(arr: np.ndarray, spacing: tuple = (0.125, 0.125), small_center_threshold: int = 30,
                            isolated_border_as_separate_instance_threshold=15):
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
    dilated_mm = np.array((0, 0))
    spacing = np.array(spacing)

    while np.sum(remaining_border) > 0:
        strel_size = [0, 0]
        maximum_dilation = max(dilated_mm)
        for i in range(2):
            if spacing[i] == min(spacing):
                strel_size[i] = 1
                continue
            if dilated_mm[i] + spacing[i] / 2 < maximum_dilation:
                strel_size[i] = 1
        ball_here = disk(1)

        if strel_size[0] == 0: ball_here = ball_here[1:2]
        if strel_size[1] == 0: ball_here = ball_here[:, 1:2]

        #print(1)
        dilated = dilation(current, ball_here)
        diff = (current == 0) & (dilated != current)
        final[diff & remaining_border] = dilated[diff & remaining_border]
        remaining_border[diff] = 0
        current = dilated
        dilated_mm = [dilated_mm[i] + spacing[i] if strel_size[i] == 1 else dilated_mm[i] for i in range(2)]

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


def load_convert_to_instance_save(file_in: str, file_out: str, spacing):
    img = sitk.ReadImage(file_in)
    img_npy = sitk.GetArrayFromImage(img)
    out = convert_to_instance_seg(img_npy[0], spacing)[None]
    out_itk = sitk.GetImageFromArray(out.astype(np.int16))
    out_itk.CopyInformation(img)
    sitk.WriteImage(out_itk, file_out)


def convert_folder_to_instanceseg(folder_in: str, folder_out: str, spacing, processes: int = 12):
    input_files = subfiles(folder_in, suffix=".nii.gz", join=False)
    maybe_mkdir_p(folder_out)
    output_files = [join(folder_out, i) for i in input_files]
    input_files = [join(folder_in, i) for i in input_files]
    p = Pool(processes)
    r = []
    for i, o in zip(input_files, output_files):
        r.append(
            p.starmap_async(
                load_convert_to_instance_save,
                ((i, o, spacing),)
            )
        )
    _ = [i.get() for i in r]
    p.close()
    p.join()


def convert_to_tiff(nifti_image: str, output_name: str):
    npy = sitk.GetArrayFromImage(sitk.ReadImage(nifti_image))
    imsave(output_name, npy[0].astype(np.uint16),  compress=6)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_train")
    parser.add_argument("--source_test")
    args = parser.parse_args()
    source_train = args.source_train
    source_test = args.source_test
    # source_train = "/home/fabian/Downloads/Fluo-N2DH-SIM+_train"
    # source_test = "/home/fabian/Downloads/Fluo-N2DH-SIM+_test"

    spacing = (0.125, 0.125)

    # adding the time information is a hassle, bear with us. We first create a dummy task under id 999, then copy it and finally put the time information in
    border_thickness = 0.7

    p = Pool(16)

    # now add the time information and make this a real task
    task_id = 89
    additional_time_steps = 4
    task_name = 'Fluo-N2DH-SIM_thickborder_time'
    foldername = 'Task%03.0d_' % task_id + task_name

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
                p.starmap_async(load_bmp_convert_to_nifti_borders_2d,
                                ((img_file, lab_file, img_out_base, anno_out, spacing, border_thickness),)))
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
                p.starmap_async(load_bmp_convert_to_nifti_borders_2d,
                                ((img_file, lab_file, img_out_base, anno_out, spacing, border_thickness),)))
            test_patient_names.append(casename)

    _ = [i.get() for i in res]
    p.close()
    p.join()

    # generate dataset.json
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

    # now add additional time information
    for fld in ['imagesTr', 'imagesTs']:
        curr = join(out_base, fld)
        for seq in ['01', '02']:
            images = subfiles(curr, prefix=seq, join=False)
            for i in images:
                current_timestep = int(i.split('_')[1][1:])
                renamed = join(curr, i.replace("_0000", "_%04.0d" % additional_time_steps))
                shutil.move(join(curr, i), renamed)
                for previous_timestep in range(-additional_time_steps, 0):
                    # previous time steps will already have been processed and renamed!
                    expected_filename = join(curr, seq + "_t%03.0d" % (
                                current_timestep + previous_timestep) + "_%04.0d" % additional_time_steps + ".nii.gz")
                    if not isfile(expected_filename):
                        # create empty image
                        img = sitk.ReadImage(renamed)
                        empty = sitk.GetImageFromArray(np.zeros_like(sitk.GetArrayFromImage(img)))
                        empty.CopyInformation(img)
                        sitk.WriteImage(empty, join(curr, i.replace("_0000", "_%04.0d" % (
                                    additional_time_steps + previous_timestep))))
                    else:
                        shutil.copy(expected_filename, join(curr, i.replace("_0000", "_%04.0d" % (
                                    additional_time_steps + previous_timestep))))
    dataset = load_json(join(out_base, 'dataset.json'))
    dataset['modality'] = {
        '0': 't_minus 4',
        '1': 't_minus 3',
        '2': 't_minus 2',
        '3': 't_minus 1',
        '4': 'frame of interest',
    }
    save_json(dataset, join(out_base, 'dataset.json'))

    # we do not need custom splits since we train on all training cases

    # test set predictions are converted to instance seg with convert_folder_to_instanceseg
    # convert_folder_to_instanceseg('/home/fabian/temp/OUTPUT_DIRECTORY_2D', '/home/fabian/temp/OUTPUT_DIRECTORY_2D_instance',
    #                               spacing, 12)

    # test set predictions are converted to tiff with convert_to_tiff
    # input_files = nifti_files('/home/fabian/temp/OUTPUT_DIRECTORY_2D_instance', join=False)
    # output_folder = '/home/fabian/temp/OUTPUT_DIRECTORY_2D_instance_tiff'
    # maybe_mkdir_p(output_folder)
    # output_files = [join(output_folder, i[:-7] + '.tif') for i in input_files]
    # input_files = [join('/home/fabian/temp/OUTPUT_DIRECTORY_2D_instance', i) for i in input_files]
    # for i, o in zip(input_files, output_files):
    #     convert_to_tiff(i, o)
