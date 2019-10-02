import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "nnUNetPlans"

    overwrite_plans = {
        'nnUNetTrainerV2_2': ["nnUNetPlans", "nnUNetPlansisoPatchesInVoxels"], # r
        'nnUNetTrainerV2': ["nnUNetPlansnonCT", "nnUNetPlansCT2", "nnUNetPlansallConv3x3",
                            "nnUNetPlansfixedisoPatchesInVoxels", "nnUNetPlanstargetSpacingForAnisoAxis",
                            "nnUNetPlanspoolBasedOnSpacing", "nnUNetPlansfixedisoPatchesInmm"]
    }

    trainers = ['nnUNetTrainer'] + ['nnUNetTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'nnUNetTrainerNewCandidate24_2',
        'nnUNetTrainerNewCandidate24_3',
        'nnUNetTrainerNewCandidate26_2',
        'nnUNetTrainerNewCandidate27_2',
        'nnUNetTrainerNewCandidate23_always3DDA',
        'nnUNetTrainerNewCandidate23_corrInit',
        'nnUNetTrainerNewCandidate23_noOversampling',
        'nnUNetTrainerNewCandidate23_softDS',
        'nnUNetTrainerNewCandidate23_softDS2',
        'nnUNetTrainerNewCandidate23_softDS3',
        'nnUNetTrainerNewCandidate23_softDS4',
        'nnUNetTrainerNewCandidate23_2_fp16',
        'nnUNetTrainerNewCandidate23_2',
        'nnUNetTrainerVer2',
        'nnUNetTrainerV2_2',
        'nnUNetTrainerV2_3',
        'nnUNetTrainerV2_3_CE_GDL',
        'nnUNetTrainerV2_3_dcTopk10',
        'nnUNetTrainerV2_3_dcTopk20',
        'nnUNetTrainerV2_3_fp16',
        'nnUNetTrainerV2_3_softDS4',
        'nnUNetTrainerV2_3_softDS4_clean',
        'nnUNetTrainerV2_3_softDS4_clean_improvedDA',
        'nnUNetTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'nnUNetTrainerV2_3_softDS4_radam',
        'nnUNetTrainerV2_3_softDS4_radam_lowerLR',

        'nnUNetTrainerV2_2_schedule',
        'nnUNetTrainerV2_2_schedule2',
        'nnUNetTrainerV2_2_clean',
        'nnUNetTrainerV2_2_clean_improvedDA_newElDef',

        'nnUNetTrainerV2_2_fixes', # running
        'nnUNetTrainerV2_BN', # running
        'nnUNetTrainerV2_noDeepSupervision', # running
        'nnUNetTrainerV2_softDeepSupervision', # running
        'nnUNetTrainerV2_noDataAugmentation', # running
        'nnUNetTrainerV2_Loss_CE', # running
        'nnUNetTrainerV2_Loss_CEGDL',
        'nnUNetTrainerV2_Loss_Dice',
        'nnUNetTrainerV2_Loss_DiceTopK10',
        'nnUNetTrainerV2_Loss_TopK10',
        'nnUNetTrainerV2_Adam', # running
        'nnUNetTrainerV2_Adam_nnUNetTrainerlr', # running
        'nnUNetTrainerV2_SGD_ReduceOnPlateau', # running
        'nnUNetTrainerV2_SGD_lr1en1', # running
        'nnUNetTrainerV2_SGD_lr1en3', # running
        'nnUNetTrainerV2_fixedNonlin', # running
        'nnUNetTrainerV2_GeLU', # running
        'nnUNetTrainerV2_3ConvPerStage',
        'nnUNetTrainerV2_NoNormalization',
        'nnUNetTrainerV2_Adam_ReduceOnPlateau',
        'nnUNetTrainerV2_fp16',
        'nnUNetTrainerV2', # see overwrite_plans
        'nnUNetTrainerV2_noMirroring',
        'nnUNetTrainerV2_momentum095',
        'nnUNetTrainerV2_momentum098',
        'nnUNetTrainerV2_warmup',
        'nnUNetTrainerV2_Loss_Dice_LR1en3',
        'nnUNetTrainerV2_NoNormalization_lr1en3',
        # 'nnUNetTrainerV2_3ConvPerStage',
        # 'nnUNetTrainerV2_3ConvPerStage',
        # 'nnUNetTrainerV2_3ConvPerStage',
        # 'nnUNetTrainerV2_3ConvPerStage',
        # 'nnUNetTrainerV2_3ConvPerStage',
        # 'nnUNetTrainerV2_3ConvPerStage',
        # 'nnUNetTrainerV2_3ConvPerStage',
        # 'nnUNetTrainerV2_3ConvPerStage',
    ]

    datasets = \
        {"Task01_BrainTumour": ("3d_fullres", ),
        "Task02_Heart": ("3d_fullres",),
        #"Task24_Promise": ("3d_fullres",),
        #"Task27_ACDC": ("3d_fullres",),
        "Task03_Liver": ("3d_fullres", "3d_lowres"),
        "Task04_Hippocampus": ("3d_fullres",),
        "Task05_Prostate": ("3d_fullres",),
        "Task06_Lung": ("3d_fullres", "3d_lowres"),
        "Task07_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task08_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task09_Spleen": ("3d_fullres", "3d_lowres"),
        "Task10_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:6]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                f.write("\n")

                if all_present:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
