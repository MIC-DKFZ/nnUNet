import numpy as np
import subprocess
from collections import OrderedDict
from nnunet.paths import splitted_4d_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from skimage import io
import SimpleITK as sitk
import shutil


if __name__ == "__main__":
    # download from here https://www.epfl.ch/labs/cvlab/data/data-em/

    base = "/media/fabian/My Book/datasets/EPFL_MITO_SEG"
    # the orientation of VerSe is all fing over the place. run fslreorient2std to correct that (hopefully!)
    # THIS CAN HAVE CONSEQUENCES FOR THE TEST SET SUBMISSION! CAREFUL!
    train_volume = io.imread(join(base, "training.tif"))
    train_labels = io.imread(join(base, "training_groundtruth.tif"))
    train_labels[train_labels == 255] = 1
    test_volume = io.imread(join(base, "testing.tif"))
    test_labels = io.imread(join(base, "testing_groundtruth.tif"))
    test_labels[test_labels == 255] = 1

    task_id = 59
    task_name = "EPFL_EM_MITO_SEG"

    foldername = "Task%02.0d_%s" % (task_id, task_name)

    out_base = join(splitted_4d_output_dir, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelste = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelste)

    img_tr_itk = sitk.GetImageFromArray(train_volume.astype(np.float32))
    lab_tr_itk = sitk.GetImageFromArray(train_labels.astype(np.uint8))
    img_te_itk = sitk.GetImageFromArray(test_volume.astype(np.float32))
    lab_te_itk = sitk.GetImageFromArray(test_labels.astype(np.uint8))

    img_tr_itk.SetSpacing((5, 5, 5))
    lab_tr_itk.SetSpacing((5, 5, 5))
    img_te_itk.SetSpacing((5, 5, 5))
    lab_te_itk.SetSpacing((5, 5, 5))

    # 5 copies, otherwise we cannot run nnunet (5 fold cv needs that)
    sitk.WriteImage(img_tr_itk, join(imagestr, "training0_0000.nii.gz"))
    shutil.copy(join(imagestr, "training0_0000.nii.gz"), join(imagestr, "training1_0000.nii.gz"))
    shutil.copy(join(imagestr, "training0_0000.nii.gz"), join(imagestr, "training2_0000.nii.gz"))
    shutil.copy(join(imagestr, "training0_0000.nii.gz"), join(imagestr, "training3_0000.nii.gz"))
    shutil.copy(join(imagestr, "training0_0000.nii.gz"), join(imagestr, "training4_0000.nii.gz"))

    sitk.WriteImage(lab_tr_itk, join(labelstr, "training0.nii.gz"))
    shutil.copy(join(labelstr, "training0.nii.gz"), join(labelstr, "training1.nii.gz"))
    shutil.copy(join(labelstr, "training0.nii.gz"), join(labelstr, "training2.nii.gz"))
    shutil.copy(join(labelstr, "training0.nii.gz"), join(labelstr, "training3.nii.gz"))
    shutil.copy(join(labelstr, "training0.nii.gz"), join(labelstr, "training4.nii.gz"))

    sitk.WriteImage(img_te_itk, join(imagests, "testing.nii.gz"))
    sitk.WriteImage(lab_te_itk, join(labelste, "testing.nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = task_name
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "EM",
    }
    json_dict['labels'] = {i: str(i) for i in range(2)}

    json_dict['numTraining'] = 5
    json_dict['numTest'] = 1
    json_dict['training'] = [{'image': "./imagesTr/training%d.nii.gz" % i, "label": "./labelsTr/training%d.nii.gz" % i} for i in
                             range(5)]
    json_dict['test'] = ["./imagesTs/testing.nii.gz"]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))