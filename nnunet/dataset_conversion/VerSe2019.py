from collections import OrderedDict
from nnunet.paths import splitted_4d_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


if __name__ == "__main__":
    base = "/media/fabian/DeepLearningData/VerSe2019"

    task_id = 56
    task_name = "VerSe"

    foldername = "Task%02.0d_%s" % (task_id, task_name)

    out_base = join(splitted_4d_output_dir, foldername)
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
