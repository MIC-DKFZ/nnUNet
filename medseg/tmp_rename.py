from medseg import utils
import os

path = "/gris/gris-f/homelv/kgotkows/datasets/covid19/MMRF/test/masks/"
filenames = utils.load_filenames(path)

index = 1
for filename in filenames:
    os.rename(filename, path + str(index).zfill(4) + "_0000.nii.gz")