from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from multiprocessing import Pool
import nibabel


def reorient(filename):
    img = nibabel.load(filename)
    img = nibabel.as_closest_canonical(img)
    nibabel.save(img, filename)


if __name__ == "__main__":
    base = "/media/fabian/DeepLearningData/Pancreas-CT"

    # reorient
    p = Pool(8)
    results = []

    for f in subfiles(join(base, "data"), suffix=".nii.gz"):
        results.append(p.map_async(reorient, (f, )))
    _ = [i.get() for i in results]

    for f in subfiles(join(base, "TCIA_pancreas_labels-02-05-2017"), suffix=".nii.gz"):
        results.append(p.map_async(reorient, (f, )))
    _ = [i.get() for i in results]

    task_id = 62
    task_name = "NIHPancreas"

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
    cases = list(range(1, 83))
    folder_data = join(base, "data")
    folder_labels = join(base, "TCIA_pancreas_labels-02-05-2017")
    for c in cases:
        casename = "pancreas_%04.0d" % c
        shutil.copy(join(folder_data, "PANCREAS_%04.0d.nii.gz" % c), join(imagestr, casename + "_0000.nii.gz"))
        shutil.copy(join(folder_labels, "label%04.0d.nii.gz" % c), join(labelstr, casename + ".nii.gz"))
        train_patient_names.append(casename)

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = task_name
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see website"
    json_dict['licence'] = "see website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Pancreas",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
