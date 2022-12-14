if __name__ == "__main__":
    # after the Nature Methods paper we switch our evaluation to a different (more stable/high quality) set of
    # datasets for evaluation and future development
    configurations_all = {
        216: (
            # "3d_lowres",
            # "3d_lowres_sparse_slicewise_10",
            # "3d_lowres_sparse_slicewise_30",
            # "3d_lowres_sparse_randblobs",
            # "3d_lowres_sparse_pixelwise",
            # '3d_lowres_sparse_hybridsparsepatchesslices',
            # '3d_lowres_sparse_sparsepatches',
            # '3d_lowres_sparse_sliceOSfg10',
            # '3d_lowres_sparse_slicewiserand10',
            # '3d_lowres_sparse_blobs',
            # '3d_lowres_sparse_sparsepatches40p',
            # '3d_lowres_sparse_hybridsparsepatchesslices40p',
            # '3d_lowres_sparse_pixelwise10',
            # '3d_lowres_sparse_pixelwise5',
            # '3d_lowres_sparse_pixelwise30',
            # '3d_lowres_sparse_randOrthSlices3',
            # '3d_lowres_sparse_randOrthSlices5',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_10',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_5',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_3',
            # '3d_lowres_sparse_pixelwise50',
            # '3d_lowres_sparse_pixelwise1',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_10',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_5_ppc025',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_3_ppc0167',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_10_ppc05',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_30_ppc1',
            # '3d_lowres_sparse_rand_ori_slices_with_oversampling_50_ppc1',
            # '3d_lowres_sparse_randblobs5',
            # '3d_lowres_sparse_randblobs10',
            # '3d_lowres_sparse_randblobs30',
            # '3d_lowres_sparse_randblobs50',
            # '3d_lowres_sparse_blobs5',
            # '3d_lowres_sparse_blobs10',
            # '3d_lowres_sparse_blobs30',
            # '3d_lowres_sparse_blobs50',
            # '3d_lowres_sparse_randOrthSlices10',
            # '3d_lowres_sparse_randOrthSlices30',
            # '3d_lowres_sparse_randOrthSlices50',
            # '3d_lowres_sparse_sparsepatches3',
            # '3d_lowres_sparse_sparsepatches5',
            # '3d_lowres_sparse_sparsepatches10',
            # '3d_lowres_sparse_sparsepatches30',
            # '3d_lowres_sparse_hybridsparsepatchesslices3',
            # '3d_lowres_sparse_hybridsparsepatchesslices5',
            # '3d_lowres_sparse_hybridsparsepatchesslices10',
            # '3d_lowres_sparse_hybridsparsepatchesslices30',
            '3d_lowres_sparse_patches_and_slices_3',
            '3d_lowres_sparse_patches_and_slices_5',
            '3d_lowres_sparse_patches_and_slices_3_2',
            '3d_lowres_sparse_patches_and_slices_10',
            '3d_lowres_sparse_patches_and_slices_30',
            '3d_lowres_sparse_patches_and_slices_50',
        ),
        # 994: (
        #     "3d_fullres",
        #     "3d_fullres_sparse_slicewise10",
        #     "3d_fullres_sparse_slicewise30",
        #     "3d_fullres_sparse_randblobs",
        #     "3d_fullres_sparse_pixelwise",
        #     '3d_fullres_sparse_hybridsparsepatchesslices',
        #     '3d_fullres_sparse_sparsepatches',
        #     '3d_fullres_sparse_sliceOSfg10',
        #     '3d_fullres_sparse_slicewiserand10',
        #     '3d_fullres_sparse_blobs'
        # ),
    }

    num_gpus = 1
    exclude_hosts = "-R \"select[hname!='e230-dgx2-2']\" -R \"select[hname!='e230-dgx2-1']\" -R \"select[hname!='e230-dgx1-1']\""
    resources = "-R \"tensorcore\""
    gpu_requirements = f"-gpu num={num_gpus}:j_exclusive=yes:mode=exclusive_process:gmem=33G"
    queue = "-q gpu-lowprio"
    preamble = "-L /bin/bash \"source ~/load_env_cluster3.sh && nnUNet_def_n_proc=24"
    train_command = 'nnUNetv2_train'

    folds = (0, )
    use_this = configurations_all
    # use_this = merge(configurations_3d_fr_only, configurations_3d_lr_only)
    # use_this = merge(use_this, configurations_3d_c_only)

    use_these_modules = {
        'nnUNetTrainer': ('nnUNetPlans',),
        # 'nnUNetTrainerDA5': ('nnUNetPlans',),
        # 'nnUNetTrainerDA5ord0': ('nnUNetPlans',),
        # 'nnUNetTrainerDA5_betterIgnoreSampling': ('nnUNetPlans',),
        # 'nnUNetTrainerDA5ord0_betterIgnoreSampling': ('nnUNetPlans',),
        'nnUNetTrainer_betterIgnoreSampling': ('nnUNetPlans',),
        'nnUNetTrainerDAOrd0': ('nnUNetPlans',),
    }

    additional_arguments = f'--disable_checkpointing -num_gpus {num_gpus}'  # ''

    output_file = "/home/isensee/deleteme.txt"
    with open(output_file, 'w') as f:
        for tr in use_these_modules.keys():
            for p in use_these_modules[tr]:
                for dataset in use_this.keys():
                    for config in use_this[dataset]:
                        for fl in folds:
                            command = f'bsub {exclude_hosts} {resources} {queue} {gpu_requirements} {preamble} {train_command} {dataset} {config} {fl} -tr {tr} -p {p}'
                            if additional_arguments is not None and len(additional_arguments) > 0:
                                command += f' {additional_arguments}'
                            f.write(f'{command}\"\n')

