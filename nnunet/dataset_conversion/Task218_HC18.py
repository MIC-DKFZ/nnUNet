from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from skimage.io import imread

from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data


def convert_case(source_folder: str, case_identifier: str, voxel_spacing: float, images_folder: str,
                 labels_folder: str = None) -> None:
    image = imread(join(source_folder, case_identifier + '.png'))
    image_itk = sitk.GetImageFromArray(image[None])
    image_itk.SetSpacing((voxel_spacing, voxel_spacing, 999))
    sitk.WriteImage(image_itk, join(images_folder, case_identifier + '_0000.nii.gz'))

    if labels_folder is not None:
        annotation = imread(join(source_folder, case_identifier + '_Annotation.png'))
        annotation[annotation > 0] = 1
        annotation = binary_fill_holes(annotation).astype(np.uint8)
        annotation_itk = sitk.GetImageFromArray(annotation[None])
        annotation_itk.SetSpacing((voxel_spacing, voxel_spacing, 999))
        sitk.WriteImage(annotation_itk, join(labels_folder, case_identifier + '.nii.gz'))


if __name__ == '__main__':
    training_data_folder = '/home/isensee/Downloads/training_set'
    training_data_csv = '/home/isensee/Downloads/training_set_pixel_size_and_HC.csv'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 218
    task_name = "HC18"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    p = Pool(18)
    r = []
    csv_content = np.loadtxt(training_data_csv, dtype=str, skiprows=1, delimiter=',')
    for filename, pixelsize, _ in csv_content:
        pixelsize = float(pixelsize)
        filename = filename[:-4]
        r.append(p.starmap_async(
            convert_case,
            ((training_data_folder, filename, pixelsize, imagestr, labelstr),)
        ))
    _ = [i.get() for i in r]

    r = []
    test_data_folder = '/home/isensee/Downloads/test_set'
    test_csv = '/home/isensee/Downloads/test_set_pixel_size.csv'
    csv_content = np.loadtxt(test_csv, dtype=str, skiprows=1, delimiter=',')
    for filename, pixelsize in csv_content:
        pixelsize = float(pixelsize)
        filename = filename[:-4]
        r.append(p.starmap_async(
            convert_case,
            ((test_data_folder, filename, pixelsize, imagests, None),)
        ))
    _ = [i.get() for i in r]

    generate_dataset_json(join(out_base, 'dataset.json'), imagestr, imagests, ('nonCT',), {0: 'background', 1: 'head'},
                          task_name)