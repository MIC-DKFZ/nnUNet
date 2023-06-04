
export nnUNet_raw='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/nnUNet_raw'
export nnUNet_preprocessed='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/preprocessed'
export nnUNet_results='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/nnUNet_trained_models'

python SSL/preprocessing/splitting_to_patches.py \
 -d='Dataset026_AIS' \
 -p='nnUNetPlansSSL' \
 -c='3d_fullres' \
 -np=8 \
 -cfg='/home/jseia/Desktop/thesis/code/nnUNet_ais/SSL/cfg_files/config_not_tbi.yml'