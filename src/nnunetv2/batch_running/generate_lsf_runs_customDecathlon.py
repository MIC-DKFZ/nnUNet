from copy import deepcopy
import numpy as np


def merge(dict1, dict2):
    keys = np.unique(list(dict1.keys()) + list(dict2.keys()))
    keys = np.unique(keys)
    res = {}
    for k in keys:
        all_configs = []
        if dict1.get(k) is not None:
            all_configs += list(dict1[k])
        if dict2.get(k) is not None:
            all_configs += list(dict2[k])
        if len(all_configs) > 0:
            res[k] = tuple(np.unique(all_configs))
    return res


if __name__ == "__main__":
    # after the Nature Methods paper we switch our evaluation to a different (more stable/high quality) set of
    # datasets for evaluation and future development
    configurations_all = {
        2: ("3d_fullres", "2d"),
        3: ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
        4: ("2d", "3d_fullres"),
        5: ("2d", "3d_fullres"),
        8: ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
        10: ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
        17: ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
        24: ("2d", "3d_fullres"),
        27: ("2d", "3d_fullres"),
        38: ("2d", "3d_fullres"),
        55: ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
        137: ("2d", "3d_fullres"),
        220: ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
        # 221: ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
        223: ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
    }

    configurations_3d_fr_only = {
        i: ("3d_fullres", ) for i in configurations_all if "3d_fullres" in configurations_all[i]
    }

    configurations_3d_c_only = {
        i: ("3d_cascade_fullres", ) for i in configurations_all if "3d_cascade_fullres" in configurations_all[i]
    }

    configurations_3d_lr_only = {
        i: ("3d_lowres", ) for i in configurations_all if "3d_lowres" in configurations_all[i]
    }

    configurations_2d_only = {
        i: ("2d", ) for i in configurations_all if "2d" in configurations_all[i]
    }

    num_gpus = 1
    exclude_hosts = "-R \"select[hname!='e230-dgx2-2']\" -R \"select[hname!='e230-dgx2-1']\""
    resources = ""
    gpu_requirements = f"-gpu num={num_gpus}:j_exclusive=yes:gmem=1G"#gmodel=NVIDIAA100_PCIE_40GB"
    queue = "-q gpu"
    preamble = "\". ~/load_env_torch224.sh && " # -L /bin/bash
    train_command = 'nnUNetv2_train'


    folds = (0, )
    # use_this = configurations_2d_only
    use_this = configurations_2d_only
    # use_this = merge(use_this, configurations_3d_c_only)

    datasets = [3, 4, 5, 8, 10, 17, 27, 55, 220, 223]
    use_this = {i: use_this[i] for i in datasets}

    use_these_modules = {
        # 'nnUNetTrainer_newSpatialAug': ('nnUNetPlans',),
        # 'nnUNetTrainerBN': ('nnUNetPlans',),
        # 'nnUNetTrainer_newSpatialAug_withElDef_noPref': ('nnUNetPlans',),
        # 'nnUNetTrainer': ('nnUNetConvNextEncUNetPlans_smallks_and_shallow',),
        # 'nnUNetTrainer_convnextenc_regularconvblock': ('nnUNetPlans_convnext', 'nnUNetConvNextEncUNetPlans_smallks_and_shallow'),
        # 'nnUNetTrainer_newSpatialAug_withElDef2': ('nnUNetPlans',),
        # 'nnUNetTrainer_newSpatialAug_withElDef3': ('nnUNetPlans',),
        # 'nnUNetTrainer_newSpatialAug_noPref': ('nnUNetPlans',),
        # 'nnUNetTrainerDiceCELoss_noSmooth': ('nnUNetPlans',),
        # 'nnUNetTrainer_DASegOrd0': ('nnUNetPlans',),
        # 'nnUNetTrainerAdamW_WDe2': ('nnUNetPlans',),
        # 'nnUNetTrainerUMambaBot': ('nnUNetPlans',),
        # 'nnUNetTrainerUMambaEnc': ('nnUNetPlans',),
        # 'nnUNetTrainer_fasterDA': ('nnUNetPlans', 'nnUNetResEncUNetLPlans'),
        # 'nnUNetTrainer_noDummy2DDA': ('nnUNetResEncUNetMPlans', ),
        'nnUNetTrainer': ('nnUNetResEncUNetMPlans', ),
        # 'nnUNetTrainer_probabilisticOversampling_033': ('nnUNetResEncUNetMPlans', ),
        # 'nnUNetTrainer_probabilisticOversampling_010': ('nnUNetResEncUNetMPlans',),
        # BN
    }

    additional_arguments = f' -num_gpus {num_gpus} --disable_checkpointing'  # ''

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

