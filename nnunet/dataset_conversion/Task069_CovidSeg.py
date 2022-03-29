import shutil

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
import SimpleITK as sitk
from nnunet.paths import nnUNet_raw_data

if __name__ == '__main__':
    #data is available at http://medicalsegmentation.com/covid19/
    download_dir = '/home/fabian/Downloads'

    task_id = 69
    task_name = "CovidSeg"

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

    # the niftis are 3d, but they are just stacks of 2d slices from different patients. So no 3d U-Net, please

    # the training stack has 100 slices, so we split it into 5 equally sized parts (20 slices each) for cross-validation
    training_data = sitk.GetArrayFromImage(sitk.ReadImage(join(download_dir, 'tr_im.nii.gz')))
    training_labels = sitk.GetArrayFromImage(sitk.ReadImage(join(download_dir, 'tr_mask.nii.gz')))

    for f in range(5):
        this_name = 'part_%d' % f
        data = training_data[f::5]
        labels = training_labels[f::5]
        sitk.WriteImage(sitk.GetImageFromArray(data), join(imagestr, this_name + '_0000.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(labels), join(labelstr, this_name + '.nii.gz'))
        train_patient_names.append(this_name)

    shutil.copy(join(download_dir, 'val_im.nii.gz'), join(imagests, 'val_im.nii.gz'))

    test_patient_names.append('val_im')

    json_dict = {}
    json_dict['name'] = task_name
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "nonct",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "stuff1",
        "2": "stuff2",
        "3": "stuff3",
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
