from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/home/fabian/data/nnUNet_preprocessed/Task004_Hippocampus/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = '/home/fabian/data/nnUNet_preprocessed/Task004_Hippocampus/nnUNetPlansv2.1_LISA_plans_3D.pkl'
    a = load_pickle(input_file)
    a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))
    save_pickle(a, output_file)