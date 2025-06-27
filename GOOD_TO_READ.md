PATHS: Set them in paths.py


"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""
os.environ['nnUNet_raw'] = '/mnt/c/Users/kfsolofsson/Documents/PhD/Code/nnUNet/nnUNetdata/nnUNET_raw'
os.environ['nnUNet_preprocessed'] = '/mnt/c/Users/kfsolofsson/Documents/PhD/Code/nnUNet/nnUNetdata/nnUNET_preprocessed'
os.environ['nnUNet_results'] = '/mnt/c/Users/kfsolofsson/Documents/PhD/Code/nnUNet/nnUNetdata/nnUNET_results'

nnUNet_raw = os.environ.get('nnUNet_raw')
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')

if nnUNet_raw is None:
    print("nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if nnUNet_preprocessed is None:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if nnUNet_results is None:
    print("nnUNet_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")


Predict / Inference run:
nnUNetv2_predict   -i "Code/nnUNet/data/output_COR"   -o "Code/nnUNet/data/pred"   -d 613   -c 3d_fullres_test   -tr nnUNetTrainer_2epochs   -p nnUNetPlans  --save_probabilities

or python Code/nnUNet/predict_seg.py 