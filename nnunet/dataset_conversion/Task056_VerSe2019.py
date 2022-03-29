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


from collections import OrderedDict
import SimpleITK as sitk
from multiprocessing.pool import Pool
from nnunet.configuration import default_num_threads
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
import shutil
from medpy import metric
import numpy as np
from nnunet.utilities.image_reorientation import reorient_all_images_in_folder_to_ras


def check_if_all_in_good_orientation(imagesTr_folder: str, labelsTr_folder: str, output_folder: str) -> None:
    maybe_mkdir_p(output_folder)
    filenames = subfiles(labelsTr_folder, suffix='.nii.gz', join=False)
    import matplotlib.pyplot as plt
    for n in filenames:
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(imagesTr_folder, n[:-7] + '_0000.nii.gz')))
        lab = sitk.GetArrayFromImage(sitk.ReadImage(join(labelsTr_folder, n)))
        assert np.all([i == j for i, j in zip(img.shape, lab.shape)])
        z_slice = img.shape[0] // 2
        img_slice = img[z_slice]
        lab_slice = lab[z_slice]
        lab_slice[lab_slice != 0] = 1
        img_slice = img_slice - img_slice.min()
        img_slice = img_slice / img_slice.max()
        stacked = np.vstack((img_slice, lab_slice))
        print(stacked.shape)
        plt.imsave(join(output_folder, n[:-7] + '.png'), stacked, cmap='gray')


def evaluate_verse_case(sitk_file_ref:str, sitk_file_test:str):
    """
    Only vertebra that are present in the reference will be evaluated
    :param sitk_file_ref:
    :param sitk_file_test:
    :return:
    """
    gt_npy = sitk.GetArrayFromImage(sitk.ReadImage(sitk_file_ref))
    pred_npy = sitk.GetArrayFromImage(sitk.ReadImage(sitk_file_test))
    dice_scores = []
    for label in range(1, 26):
        mask_gt = gt_npy == label
        if np.sum(mask_gt) > 0:
            mask_pred = pred_npy == label
            dc = metric.dc(mask_pred, mask_gt)
        else:
            dc = np.nan
        dice_scores.append(dc)
    return dice_scores


def evaluate_verse_folder(folder_pred, folder_gt, out_json="/home/fabian/verse.json"):
    p = Pool(default_num_threads)
    files_gt_bare = subfiles(folder_gt, join=False)
    assert all([isfile(join(folder_pred, i)) for i in files_gt_bare]), "some files are missing in the predicted folder"
    files_pred = [join(folder_pred, i) for i in files_gt_bare]
    files_gt = [join(folder_gt, i) for i in files_gt_bare]

    results = p.starmap_async(evaluate_verse_case, zip(files_gt, files_pred))

    results = results.get()

    dct = {i: j for i, j in zip(files_gt_bare, results)}

    results_stacked = np.vstack(results)
    results_mean = np.nanmean(results_stacked, 0)
    overall_mean = np.nanmean(results_mean)

    save_json((dct, list(results_mean), overall_mean), out_json)
    p.close()
    p.join()


def print_unique_labels_and_their_volumes(image: str, print_only_if_vol_smaller_than: float = None):
    img = sitk.ReadImage(image)
    voxel_volume = np.prod(img.GetSpacing())
    img_npy = sitk.GetArrayFromImage(img)
    uniques = [i for i in np.unique(img_npy) if i != 0]
    volumes = {i: np.sum(img_npy == i) * voxel_volume for i in uniques}
    print('')
    print(image.split('/')[-1])
    print('uniques:', uniques)
    for k in volumes.keys():
        v = volumes[k]
        if print_only_if_vol_smaller_than is not None and v > print_only_if_vol_smaller_than:
            pass
        else:
            print('k:', k, '\tvol:', volumes[k])


def remove_label(label_file: str, remove_this: int, replace_with: int = 0):
    img = sitk.ReadImage(label_file)
    img_npy = sitk.GetArrayFromImage(img)
    img_npy[img_npy == remove_this] = replace_with
    img2 = sitk.GetImageFromArray(img_npy)
    img2.CopyInformation(img)
    sitk.WriteImage(img2, label_file)


if __name__ == "__main__":
    ### First we create a nnunet dataset from verse. After this the images will be all willy nilly in their
    # orientation because that's how VerSe comes
    #base = '/media/fabian/DeepLearningData/VerSe2019'
    #base = "/home/fabian/data/VerSe2019"
    base = "D:/Work/Issues/Meeting/Python/Data/the MICCAI-challenge/VerSe 2019 (MICCAI challenge data structure)"

    # correct orientation
    train_files_base = subfiles(join(base, "train"), join=False, suffix="_seg.nii.gz")
    train_segs = [i[:-len("_seg.nii.gz")] + "_seg.nii.gz" for i in train_files_base]
    train_data = [i[:-len("_seg.nii.gz")] + ".nii.gz" for i in train_files_base]
    test_files_base = [i[:-len(".nii.gz")] for i in subfiles(join(base, "test"), join=False, suffix=".nii.gz")]
    test_data = [i + ".nii.gz" for i in test_files_base]

    task_id = 56
    task_name = "VerSe"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = [i[:-len("_seg.nii.gz")] for i in subfiles(join(base, "train"), join=False, suffix="_seg.nii.gz")]
    for p in train_patient_names:
        curr = join(base, "train")
        label_file = join(curr, p + "_seg.nii.gz")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))

    test_patient_names = [i[:-7] for i in subfiles(join(base, "test"), join=False, suffix=".nii.gz")]
    for p in test_patient_names:
        curr = join(base, "test")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "VerSe2019"
    json_dict['description'] = "VerSe2019"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {i: str(i) for i in range(26)}

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))

    # now we reorient all those images to ras. This saves a pkl with the original affine. We need this information to
    # bring our predictions into the same geometry for submission
    reorient_all_images_in_folder_to_ras(imagestr)
    reorient_all_images_in_folder_to_ras(imagests)
    reorient_all_images_in_folder_to_ras(labelstr)

    # sanity check
    check_if_all_in_good_orientation(imagestr, labelstr, join(out_base, 'sanitycheck'))
    # looks good to me - proceed

    # check the volumes of the vertebrae
    _ = [print_unique_labels_and_their_volumes(i, 1000) for i in subfiles(labelstr, suffix='.nii.gz')]

    # some cases appear fishy. For example, verse063.nii.gz has labels [1, 20, 21, 22, 23, 24] and 1 only has a volume
    # of 63mm^3

    #let's correct those

    # 19 is connected to the image border and should not be segmented. Only one slice of 19 is segmented in the
    # reference. Looks wrong
    remove_label(join(labelstr, 'verse031.nii.gz'), 19, 0)

    # spurious annotation of 18 (vol: 8.00)
    remove_label(join(labelstr, 'verse060.nii.gz'), 18, 0)

    # spurious annotation of 16 (vol: 3.00)
    remove_label(join(labelstr, 'verse061.nii.gz'), 16, 0)

    # spurious annotation of 1 (vol: 63.00) although the rest of the vertebra is [20, 21, 22, 23, 24]
    remove_label(join(labelstr, 'verse063.nii.gz'), 1, 0)

    # spurious annotation of 3 (vol: 9.53) although the rest of the vertebra is
    # [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    remove_label(join(labelstr, 'verse074.nii.gz'), 3, 0)

    # spurious annotation of 3 (vol: 15.00)
    remove_label(join(labelstr, 'verse097.nii.gz'), 3, 0)

    # spurious annotation of 3 (vol: 10) although the rest of the vertebra is
    # [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    remove_label(join(labelstr, 'verse151.nii.gz'), 3, 0)

    # spurious annotation of 25 (vol: 4) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    remove_label(join(labelstr, 'verse201.nii.gz'), 25, 0)

    # spurious annotation of 23 (vol: 8) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8]
    remove_label(join(labelstr, 'verse207.nii.gz'), 23, 0)

    # spurious annotation of 23 (vol: 12) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    remove_label(join(labelstr, 'verse208.nii.gz'), 23, 0)

    # spurious annotation of 23 (vol: 2) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    remove_label(join(labelstr, 'verse212.nii.gz'), 23, 0)

    # spurious annotation of 20 (vol: 4) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    remove_label(join(labelstr, 'verse214.nii.gz'), 20, 0)

    # spurious annotation of 23 (vol: 15) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8]
    remove_label(join(labelstr, 'verse223.nii.gz'), 23, 0)

    # spurious annotation of 23 (vol: 1) and 25 (vol: 7) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    remove_label(join(labelstr, 'verse226.nii.gz'), 23, 0)
    remove_label(join(labelstr, 'verse226.nii.gz'), 25, 0)

    # spurious annotation of 25 (vol: 27) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8]
    remove_label(join(labelstr, 'verse227.nii.gz'), 25, 0)

    # spurious annotation of 20 (vol: 24) although the rest of the vertebra is
    # [1, 2, 3, 4, 5, 6, 7, 8]
    remove_label(join(labelstr, 'verse232.nii.gz'), 20, 0)


    # Now we are ready to run nnU-Net


    """# run this part of the code once training is done
    folder_gt = "/media/fabian/My Book/MedicalDecathlon/nnUNet_raw_splitted/Task056_VerSe/labelsTr"

    folder_pred = "/home/fabian/drives/datasets/results/nnUNet/3d_fullres/Task056_VerSe/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_raw"
    out_json = "/home/fabian/Task056_VerSe_3d_fullres_summary.json"
    evaluate_verse_folder(folder_pred, folder_gt, out_json)

    folder_pred = "/home/fabian/drives/datasets/results/nnUNet/3d_lowres/Task056_VerSe/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_raw"
    out_json = "/home/fabian/Task056_VerSe_3d_lowres_summary.json"
    evaluate_verse_folder(folder_pred, folder_gt, out_json)

    folder_pred = "/home/fabian/drives/datasets/results/nnUNet/3d_cascade_fullres/Task056_VerSe/nnUNetTrainerV2CascadeFullRes__nnUNetPlansv2.1/cv_niftis_raw"
    out_json = "/home/fabian/Task056_VerSe_3d_cascade_fullres_summary.json"
    evaluate_verse_folder(folder_pred, folder_gt, out_json)"""

