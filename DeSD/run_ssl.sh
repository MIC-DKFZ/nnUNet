#!/bin/bash

export nnUNet_raw='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/nnUNet_raw'
export nnUNet_preprocessed='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/preprocessed'
export nnUNet_results='/home/jseia/Desktop/thesis/code/nnUNet_ais/nnunetv2/nnUNet_trained_models'

python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
    -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/all_ncct/config_all_ncct_001.yml'

python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
    -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/tbi/config_tbi_000.yml'

python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
    -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/non_tbi/config_non_tbi_003.yml'

python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
    -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/tbi/config_tbi_001.yml'

python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
    -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/all_ncct/config_all_ncct_004.yml'

python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
    -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/all_ncct/config_all_ncct_002.yml'

python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
    -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/non_tbi/config_non_tbi_004.yml'

python /home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/main_DeSD_ssl.py \
    -cfg '/home/jseia/Desktop/thesis/code/nnUNet_ais/DeSD/cfg_files/ssl_pretrain/tbi/config_tbi_002.yml'

