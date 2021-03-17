from medseg import utils
import os

path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task072_allGuided_ggo/guiding_masks/"
index = 110

filenames = utils.load_filenames(path)
for filename in filenames:
    os.rename(filename, filename[:-7] + "_tmp.nii.gz")

filenames = utils.load_filenames(path)
for filename in filenames:
    os.rename(filename, path + str(index).zfill(4) + "_0001.nii.gz")  # _0000
    index += 1