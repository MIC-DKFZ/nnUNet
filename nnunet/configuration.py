import os
import psutil

default_num_threads = max(1, psutil.cpu_count(logical=False) - 1)
if 'nnUNet_def_n_proc' in os.environ:
    default_num_threads = int(os.environ['nnUNet_def_n_proc'])
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  # determines what threshold to use for resampling the low resolution axis
# separately (with NN)
