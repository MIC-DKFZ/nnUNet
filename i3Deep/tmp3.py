import numpy as np
import os
from natsort import natsorted
from shutil import copyfile
from tqdm import tqdm

load_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Task148_RibSeg/labelsTs/"

filenames = os.listdir(load_dir)
filenames = np.asarray(filenames)
filenames = natsorted(filenames)

for filename in filenames:
    new_filename = filename[:-12] + ".nii.gz"
    os.rename(load_dir + filename, load_dir + new_filename)