from collections import OrderedDict
from nnunet.paths import splitted_4d_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


if __name__ == "__main__":
    base = "/media/fabian/DeepLearningData/SegTHOR"

    task_id = 55
    task_name = "SegTHOR"

    foldername = "Task%02.0d_%s" % (task_id, task_name)

    out_base = join(splitted_4d_output_dir, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    train_patients = subfolders(join(base, "train"), join=False)
    for p in train_patients:
        curr = join(base, "train", p)
        label_file = join(curr, "GT.nii.gz")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    test_patients = subfiles(join(base, "test"), join=False, suffix=".nii.gz")
    for p in test_patients:
        p = p[:-7]
        curr = join(base, "test")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        test_patient_names.append(p)


    json_dict = OrderedDict()
    json_dict['name'] = "SegTHOR"
    json_dict['description'] = "SegTHOR"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "esophagus",
        "2": "heart",
        "3": "trachea",
        "4": "aorta",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
