import numpy as np
import subprocess
from collections import OrderedDict
from nnunet.paths import splitted_4d_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from skimage import io
import SimpleITK as sitk


if __name__ == "__main__":
    # download from here http://brainiac2.mit.edu/isbi_challenge/downloads

    base = "/media/fabian/My Book/datasets/ISBI_EM_SEG"
    # the orientation of VerSe is all fing over the place. run fslreorient2std to correct that (hopefully!)
    # THIS CAN HAVE CONSEQUENCES FOR THE TEST SET SUBMISSION! CAREFUL!
    train_volume = io.imread(join(base, "train-volume.tif"))
    train_labels = io.imread(join(base, "train-labels.tif"))
    train_labels[train_labels == 255] = 1
    test_volume = io.imread(join(base, "test-volume.tif"))

    task_id = 58
    task_name = "ISBI_EM_SEG"

    foldername = "Task%02.0d_%s" % (task_id, task_name)

    out_base = join(splitted_4d_output_dir, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    img_tr_itk = sitk.GetImageFromArray(train_volume.astype(np.float32))
    lab_tr_itk = sitk.GetImageFromArray(1 - train_labels) # walls are foreground, cells background
    img_te_itk = sitk.GetImageFromArray(test_volume.astype(np.float32))

    img_tr_itk.SetSpacing((4, 4, 50))
    lab_tr_itk.SetSpacing((4, 4, 50))
    img_te_itk.SetSpacing((4, 4, 50))

    # 5 copies, otherwise we cannot run nnunet (5 fold cv needs that)
    sitk.WriteImage(img_tr_itk, join(imagestr, "training0_0000.nii.gz"))
    sitk.WriteImage(img_tr_itk, join(imagestr, "training1_0000.nii.gz"))
    sitk.WriteImage(img_tr_itk, join(imagestr, "training2_0000.nii.gz"))
    sitk.WriteImage(img_tr_itk, join(imagestr, "training3_0000.nii.gz"))
    sitk.WriteImage(img_tr_itk, join(imagestr, "training4_0000.nii.gz"))

    sitk.WriteImage(lab_tr_itk, join(labelstr, "training0.nii.gz"))
    sitk.WriteImage(lab_tr_itk, join(labelstr, "training1.nii.gz"))
    sitk.WriteImage(lab_tr_itk, join(labelstr, "training2.nii.gz"))
    sitk.WriteImage(lab_tr_itk, join(labelstr, "training3.nii.gz"))
    sitk.WriteImage(lab_tr_itk, join(labelstr, "training4.nii.gz"))

    sitk.WriteImage(img_te_itk, join(imagests, "testing.nii.gz"))

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