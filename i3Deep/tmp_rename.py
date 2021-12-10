from i3Deep import utils
import os

path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task001_BrainTumour/labelsTr/"
index = 0
keep_basename = True
modality_number = "0000"

filenames = utils.load_filenames(path)
for filename in filenames:
    os.rename(filename, filename[:-7] + "_tmp.nii.gz")

filenames = utils.load_filenames(path)
for filename in filenames:
    if keep_basename:
        name = path + os.path.basename(filename)[:-11] + "_" + modality_number + ".nii.gz"
    else:
        name = path + str(index).zfill(4) + "_" + modality_number + ".nii.gz"
        index += 1
    os.rename(filename, name)
