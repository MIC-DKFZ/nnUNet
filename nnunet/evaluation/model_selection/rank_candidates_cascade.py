import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary_cascade.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "nnUNetPlansv2.1"

    overwrite_plans = {
        'nnUNetTrainerCascadeFullRes': ['nnUNetPlans'],
    }

    trainers = [
        'nnUNetTrainerCascadeFullRes',
        'nnUNetTrainerV2CascadeFullRes_EducatedGuess',
        'nnUNetTrainerV2CascadeFullRes_EducatedGuess2',
        'nnUNetTrainerV2CascadeFullRes_EducatedGuess3',
        'nnUNetTrainerV2CascadeFullRes_lowerLR',
        'nnUNetTrainerV2CascadeFullRes',
        'nnUNetTrainerV2CascadeFullRes_noConnComp',
        'nnUNetTrainerV2CascadeFullRes_shorter_lowerLR',
        'nnUNetTrainerV2CascadeFullRes_shorter',
        'nnUNetTrainerV2CascadeFullRes_smallerBinStrel',
        #'',
        #'',
        #'',
        #'',
        #'',
        #'',
    ]

    datasets = \
        {
        "Task003_Liver": ("3d_cascade_fullres", ),
        "Task006_Lung": ("3d_cascade_fullres", ),
        "Task007_Pancreas": ("3d_cascade_fullres", ),
        "Task008_HepaticVessel": ("3d_cascade_fullres", ),
        "Task009_Spleen": ("3d_cascade_fullres", ),
        "Task010_Colon": ("3d_cascade_fullres", ),
        "Task017_AbdominalOrganSegmentation": ("3d_cascade_fullres", ),
        #"Task029_LITS": ("3d_cascade_fullres", ),
        "Task048_KiTS_clean": ("3d_cascade_fullres", ),
        "Task055_SegTHOR": ("3d_cascade_fullres", ),
        "Task056_VerSe": ("3d_cascade_fullres", ),
        #"": ("3d_cascade_fullres", ),
        }

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
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
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
