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
from collections import OrderedDict
from copy import deepcopy
from multiprocessing.pool import Pool

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.Task056_VerSe2019 import check_if_all_in_good_orientation, \
    print_unique_labels_and_their_volumes
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.image_reorientation import reorient_all_images_in_folder_to_ras


def manually_change_plans():
    pp_out_folder = join(preprocessing_output_dir, "Task083_VerSe2020")
    original_plans = join(pp_out_folder, "nnUNetPlansv2.1_plans_3D.pkl")
    assert isfile(original_plans)
    original_plans = load_pickle(original_plans)

    # let's change the network topology for lowres and fullres
    new_plans = deepcopy(original_plans)
    stages = len(new_plans['plans_per_stage'])
    for s in range(stages):
        new_plans['plans_per_stage'][s]['patch_size'] = (224, 160, 160)
        new_plans['plans_per_stage'][s]['pool_op_kernel_sizes'] = [[2, 2, 2],
                                                                   [2, 2, 2],
                                                                   [2, 2, 2],
                                                                   [2, 2, 2],
                                                                   [2, 2, 2]] # bottleneck of 7x5x5
        new_plans['plans_per_stage'][s]['conv_kernel_sizes'] = [[3, 3, 3],
                                                                [3, 3, 3],
                                                                [3, 3, 3],
                                                                [3, 3, 3],
                                                                [3, 3, 3],
                                                                [3, 3, 3]]
    save_pickle(new_plans, join(pp_out_folder, "custom_plans_3D.pkl"))


if __name__ == "__main__":
    ### First we create a nnunet dataset from verse. After this the images will be all willy nilly in their
    # orientation because that's how VerSe comes
    base = '/home/fabian/Downloads/osfstorage-archive/'

    task_id = 83
    task_name = "VerSe2020"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []

    for t in subdirs(join(base, 'training_data'), join=False):
        train_patient_names_here = [i[:-len("_seg.nii.gz")] for i in
                                    subfiles(join(base, "training_data", t), join=False, suffix="_seg.nii.gz")]
        for p in train_patient_names_here:
            curr = join(base, "training_data", t)
            label_file = join(curr, p + "_seg.nii.gz")
            image_file = join(curr, p + ".nii.gz")
            shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
            shutil.copy(label_file, join(labelstr, p + ".nii.gz"))

        train_patient_names += train_patient_names_here

    json_dict = OrderedDict()
    json_dict['name'] = "VerSe2020"
    json_dict['description'] = "VerSe2020"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {i: str(i) for i in range(29)}

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = []
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i
        in
        train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in []]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))

    # now we reorient all those images to ras. This saves a pkl with the original affine. We need this information to
    # bring our predictions into the same geometry for submission
    reorient_all_images_in_folder_to_ras(imagestr, 16)
    reorient_all_images_in_folder_to_ras(imagests, 16)
    reorient_all_images_in_folder_to_ras(labelstr, 16)

    # sanity check
    check_if_all_in_good_orientation(imagestr, labelstr, join(out_base, 'sanitycheck'))
    # looks good to me - proceed

    # check the volumes of the vertebrae
    p = Pool(6)
    _ = p.starmap(print_unique_labels_and_their_volumes, zip(subfiles(labelstr, suffix='.nii.gz'), [1000] * 113))

    # looks good

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
