#!/bin/bash

export nnUNet_raw='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/nnUNet_raw'
export nnUNet_preprocessed='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/preprocessed'
export nnUNet_results='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/nnUNet_trained_models'

# python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
#     -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/all_ncct/config_all_ncct_001.yml'

# python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
#     -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/tbi/config_tbi_000.yml'

# python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
#     -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/non_tbi/config_non_tbi_003.yml'

# python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
#     -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/tbi/config_tbi_001.yml'

# python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
#     -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/all_ncct/config_all_ncct_004.yml'

# python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
#     -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/all_ncct/config_all_ncct_002.yml'

# python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
#     -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/non_tbi/config_non_tbi_004.yml'

# python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
#     -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/tbi/config_tbi_002.yml'

# splits="/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/preprocessed/Dataset043_AIS/raw_splits_final.json"
# out_path="/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/nnUNet_trained_models/Dataset043_AIS_r1/nnUNetTrainerCfg__nnUNetPlansSSL__3d_fullres/fold_0/"
# m_path="/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/nnUNet_trained_models/Dataset043_AIS_r1/nnUNetTrainerCfg__nnUNetPlansSSL__3d_fullres"
# for i in best
# do
#     nnUNetv2_predict_from_modelfolder -i "${splits}" -o "${out_path}/validation_${i}" -m "${m_path}" \
#         -f 0 --save_probabilities -chk "checkpoint_${i}.pth" -npp 3 -nps 3 -device 'cuda'
# done