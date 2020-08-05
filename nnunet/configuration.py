import os

default_num_threads = 8 if 'nnUNet_def_n_proc' not in os.environ else os.environ['nnUNet_def_n_proc'],
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  # determines what threshold to use for resampling the low resolution axis
# separately (with NN)