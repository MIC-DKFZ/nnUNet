import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import splitted_4d_output_dir


if __name__ == "__main__":
    """
    This is the KiTS dataset after Nick fixed all the labels that had errors. Downloaded on Jan 6th 2020    
    """

    base = "/media/fabian/My Book/datasets/KiTS_clean/kits19/data"

    task_id = 64
    task_name = "KiTS_labelsFixed"

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
    all_cases = subfolders(base, join=False)

    train_patients = all_cases[:210]
    test_patients = all_cases[210:]

    for p in train_patients:
        curr = join(base, p)
        label_file = join(curr, "segmentation.nii.gz")
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    for p in test_patients:
        curr = join(base, p)
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        test_patient_names.append(p)

    json_dict = {}
    json_dict['name'] = "KiTS"
    json_dict['description'] = "kidney and kidney tumor segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "KiTS data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Kidney",
        "2": "Tumor"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
