if __name__ == '__main__':
    """
    This code probably only works within the DKFZ infrastructure (using LSF). You will need to adapt it to your scheduler! 
    """
    gpu_models = [#'NVIDIAA100_PCIE_40GB', 'NVIDIAGeForceRTX2080Ti', 'NVIDIATITANRTX', 'TeslaV100_SXM2_32GB',
                  'NVIDIAA100_SXM4_40GB']#, 'TeslaV100_PCIE_32GB']
    datasets = [2, 3, 4, 5]
    trainers = ['nnUNetTrainerBenchmark_5epochs', 'nnUNetTrainerBenchmark_5epochs_noDataLoading']
    plans = ['nnUNetPlans']
    configs = ['2d', '2d_bs3x', '2d_bs6x', '3d_fullres', '3d_fullres_bs3x', '3d_fullres_bs6x']
    num_gpus = 1

    benchmark_configurations = {d: configs for d in datasets}

    exclude_hosts = "-R \"select[hname!='e230-dgxa100-1']'\""
    resources = "-R \"tensorcore\""
    queue = "-q gpu"
    preamble = "-L /bin/bash \"source ~/load_env_torch210.sh && "
    train_command = 'nnUNet_compile=False nnUNet_results=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake_benchmark nnUNetv2_train'

    folds = (0, )

    use_these_modules = {
        tr: plans for tr in trainers
    }

    additional_arguments = f' -num_gpus {num_gpus}'  # ''

    output_file = "/home/isensee/deleteme.txt"
    with open(output_file, 'w') as f:
        for g in gpu_models:
            gpu_requirements = f"-gpu num={num_gpus}:j_exclusive=yes:gmodel={g}"
            for tr in use_these_modules.keys():
                for p in use_these_modules[tr]:
                    for dataset in benchmark_configurations.keys():
                        for config in benchmark_configurations[dataset]:
                            for fl in folds:
                                command = f'bsub {exclude_hosts} {resources} {queue} {gpu_requirements} {preamble} {train_command} {dataset} {config} {fl} -tr {tr} -p {p}'
                                if additional_arguments is not None and len(additional_arguments) > 0:
                                    command += f' {additional_arguments}'
                                f.write(f'{command}\"\n')