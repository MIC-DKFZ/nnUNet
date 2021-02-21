from medseg import utils
import os

path = "/gris/gris-f/homelv/kgotkows/datasets/covid19/MMRF/test/masks/"
filenames = utils.load_filenames(path)

index = 84
for filename in filenames:
    if "covid" in os.path.basename(filename[:-7]):  # medseg: len(os.path.basename(filename[:-7])) == 1, all other like: "study" in os.path.basename(filename[:-7])
        os.rename(filename, path + str(index).zfill(4) + "_0000.nii.gz")
        index += 1