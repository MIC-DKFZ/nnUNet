import shutil
from multiprocessing.pool import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import preprocessing_output_dir
from nnunet.training.dataloading.dataset_loading import delete_npy
from scipy.ndimage import distance_transform_edt


def compute_edts(segmentation, classes):
    res = np.zeros((len(classes), *segmentation.shape))
    for i, c in enumerate(classes):
        posmask = segmentation == c
        negmask = ~posmask
        res[i] = distance_transform_edt(negmask) * negmask - (distance_transform_edt(posmask) - 1) * posmask
    return res


def process_case(case, task_name, data_dir, classes):
    data = np.load(join(preprocessing_output_dir, task_name, data_dir, case + ".npz"))['data']
    seg = data[-1]
    seg_tmp = np.copy(seg)
    seg_tmp[seg_tmp == -1] = 0
    edts = compute_edts(seg_tmp, classes)
    out_file = join(preprocessing_output_dir, task_name, data_dir, case + ".npz")
    tmp = np.vstack((data[:-1], edts, seg[None]))
    print(case, tmp.shape)
    np.savez_compressed(out_file, data=tmp)


def add_distance_transforms(task_name, data_dir, classes, num_processes):
    cases_tr = [i[:-4] for i in subfiles(join(preprocessing_output_dir, task_name, data_dir), suffix=".npz", join=False)]
    p = Pool(num_processes)
    _ = p.starmap(process_case, zip(cases_tr, [task_name] * len(cases_tr), [data_dir] * len(cases_tr), [classes]*len(cases_tr)))
    p.close()
    p.join()


if __name__ == "__main__":
    """
    We select a source task (without ETD) and add ETD maps to the modalities
    The input_task must already have been preprocessed!
    We create a new task so that we don't overwrite the old one!
    """
    input_task = "Task02_Heart"
    output_task = "Task45_HeartWithEDT"
    data_dir = "nnUNet_stage0" # this corresponds to 3d_fullres for Task02. Needs to be adapted accordingly
    num_processes = 8

    if isdir(join(preprocessing_output_dir, output_task)):
        shutil.rmtree(join(preprocessing_output_dir, output_task))

    shutil.copytree(join(preprocessing_output_dir, input_task), join(preprocessing_output_dir, output_task))

    delete_npy(join(preprocessing_output_dir, input_task, data_dir))

    dataset_json = load_json(join(preprocessing_output_dir, output_task, "dataset.json"))
    classes = [int(i) for i in dataset_json['labels'].keys()]

    add_distance_transforms(output_task, data_dir, classes, num_processes=8)
