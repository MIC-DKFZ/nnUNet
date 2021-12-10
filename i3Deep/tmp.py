import numpy as np
import os
from natsort import natsorted
from shutil import copyfile
from tqdm import tqdm

label_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/original/RibSeg/labelsTs/"
load_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Task147_RibFrac/imagesTs/"
save_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/original/RibSeg/imagesTs/"

filenames = os.listdir(label_dir)
filenames = np.asarray(filenames)
filenames = natsorted(filenames)

# for filename in filenames:
#     new_filename = "RibFrac_" + str(int(filename[7:-15])).zfill(4) + "_0000.nii.gz"
#     os.rename(load_dir + filename, load_dir + new_filename)

for filename in tqdm(filenames):
    copyfile(load_dir + filename, save_dir + filename)