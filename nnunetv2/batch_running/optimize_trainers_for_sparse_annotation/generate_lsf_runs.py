if __name__ == "__main__":
    # after the Nature Methods paper we switch our evaluation to a different (more stable/high quality) set of
    # datasets for evaluation and future development
    configurations_all = {
        216: (
            "3d_lowres_sparse_patches_and_slices_10_2",
            "3d_lowres_sparse_randOrthSlices10",
            "3d_lowres_sparse_pixelwise10",
            "3d_lowres_sparse_patches_10_2"
        ),
    }

    num_gpus = 1
    # [f'lsf22-gpu{i:2d}' for i in range(1, 9)] + \
    exclude_hosts = [f'e230-dgx2-{i}' for i in range(1, 3)] + \
                    ['e230-dgx1-1'] + \
                    [f'e230-dgxa100-{i}' for i in range(1, 5)]
    exclude_hosts_string = ""
    for e in exclude_hosts:
        exclude_hosts_string += f"-R \"select[hname!='{e}']\" "
    resources = "-R \"tensorcore\""
    gpu_requirements = f"-gpu num={num_gpus}:j_exclusive=yes:gmem=33G"
    queue = "-q gpu-lowprio"
    preamble = "-L /bin/bash \"source ~/load_env_cluster3.sh && nnUNet_def_n_proc=24 "
    train_command = 'nnUNetv2_train'

    folds = (0, 1, 2)
    use_this = configurations_all
    # use_this = merge(configurations_3d_fr_only, configurations_3d_lr_only)
    # use_this = merge(use_this, configurations_3d_c_only)

    use_these_modules = {
        'nnUNetTrainer': ('nnUNetPlans', ),
        'nnUNetTrainer_betterIgnoreSampling': ('nnUNetPlans', ),
        'nnUNetTrainer_DASegOrd0': ('nnUNetPlans', ),
        'nnUNetTrainer_ignoreLabel_fixCEAggr': ('nnUNetPlans', ),
        'nnUNetTrainer_ignoreLabel_lossBalancing': ('nnUNetPlans',),
    }

    additional_arguments = f'--disable_checkpointing -num_gpus {num_gpus}'  # ''

    output_file = "/home/isensee/deleteme.txt"
    with open(output_file, 'w') as f:
        for tr in use_these_modules.keys():
            for p in use_these_modules[tr]:
                for dataset in use_this.keys():
                    for config in use_this[dataset]:
                        for fl in folds:
                            command = f'bsub {exclude_hosts_string} {resources} {queue} {gpu_requirements} {preamble} {train_command} {dataset} {config} {fl} -tr {tr} -p {p}'
                            if additional_arguments is not None and len(additional_arguments) > 0:
                                command += f' {additional_arguments}'
                            f.write(f'{command}\"\n')
