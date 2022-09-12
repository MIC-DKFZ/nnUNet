from typing import Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.evaluation.evaluate_predictions import load_summary_json
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name, convert_dataset_name_to_id
from nnunetv2.utilities.file_path_utilities import get_output_folder


def collect_results(trainers: dict, datasets: List, output_file: str):
    results_dirs = (nnUNet_results, )
    datasets_names = [maybe_convert_to_dataset_name(i) for i in datasets]
    configurations = ("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres")
    folds = tuple(np.arange(5))
    with open(output_file, 'w') as f:
        for i, d in zip(datasets, datasets_names):
            for c in configurations:
                for module in trainers.keys():
                    for plans in trainers[module]:
                        for r in results_dirs:
                            expected_output_folder = get_output_folder(d, module, plans, c)
                            if isdir(expected_output_folder):
                                results_folds = []
                                f.write("%s,%s,%s,%s,%s" % (d, c, module, plans, r))
                                for fl in folds:
                                    expected_output_folder_fold = get_output_folder(d, module, plans, c, fl)
                                    expected_summary_file = join(expected_output_folder_fold, "validation", "summary.json")
                                    if not isfile(expected_summary_file):
                                        print('expected output file not found:', expected_summary_file)
                                        f.write(",")
                                    else:
                                        foreground_mean = load_summary_json(expected_summary_file)['foreground_mean']['Dice']
                                        results_folds.append(foreground_mean)
                                        f.write(",%02.4f" % foreground_mean)
                                f.write(",%02.4f\n" % np.mean(results_folds))


def summarize(input_file, output_file, folds: Tuple[int, ...], configs: Tuple[str, ...], datasets, trainers):
    txt = np.loadtxt(input_file, dtype=str, delimiter=',')
    num_folds = txt.shape[1] - 6
    valid_configs = {}
    for d in datasets:
        if isinstance(d, int):
            d = maybe_convert_to_dataset_name(d)
        configs_in_txt = np.unique(txt[:, 1][txt[:, 0] == d])
        valid_configs[d] = [i for i in configs_in_txt if i in configs]
    assert max(folds) < num_folds

    with open(output_file, 'w') as f:
        f.write("name")
        for d in valid_configs.keys():
            for c in valid_configs[d]:
                f.write(",%d_%s" % (convert_dataset_name_to_id(d), c[:4]))
        f.write(',mean\n')
        valid_entries = txt[:, 4] == nnUNet_results
        for t in trainers.keys():
            trainer_locs = valid_entries & (txt[:, 2] == t)
            for pl in trainers[t]:
                f.write("%s__%s" % (t, pl))
                trainer_plan_locs = trainer_locs & (txt[:, 3] == pl)
                r = []
                for d in valid_configs.keys():
                    trainer_plan_d_locs = trainer_plan_locs & (txt[:, 0] == d)
                    for v in valid_configs[d]:
                        trainer_plan_d_config_locs = trainer_plan_d_locs & (txt[:, 1] == v)
                        if np.any(trainer_plan_d_config_locs):
                            # we cannot have more than one row
                            assert np.sum(trainer_plan_d_config_locs) == 1

                            # now check that we have all folds
                            selected_row = txt[np.argwhere(trainer_plan_d_config_locs)[0,0]]

                            fold_results = selected_row[[i + 5 for i in folds]]

                            if '' in fold_results:
                                print('missing fold in', t, pl, d, v)
                                f.write(",nan")
                                r.append(np.nan)
                            else:
                                mean_dice = np.mean([float(i) for i in fold_results])
                                f.write(",%02.4f" % mean_dice)
                                r.append(mean_dice)
                        else:
                            print('missing:', t, pl, d, v)
                            f.write(",nan")
                            r.append(np.nan)
                f.write(",%02.4f\n" % np.mean(r))


if __name__ == '__main__':
    use_these_trainers = {
        'nnUNetModule': ('nnUNetPlans', ),  # lightning variant
        'nnUNetTrainer': ('nnUNetPlans', 'nnUNetResEncUNetPlans'),  # my trainer
        'nnUNetTrainer_switchToDiceep800': ('nnUNetPlans',),
        'nnUNetTrainer_switchToDiceep100': ('nnUNetPlans',),
        'nnUNetTrainer_switchToDiceep100noSmooth': ('nnUNetPlans',),
        'nnUNetTrainer_DiceUseClip_noSmooth': ('nnUNetPlans',),
        'nnUNetTrainer_DiceUseClip': ('nnUNetPlans',),
        'nnUNetTrainer_probabilisticOversampling': ('nnUNetPlans',),
        'nnUNetTrainer_probabilisticOversampling_033': ('nnUNetPlans',),
        'nnUNetTrainer_probabilisticOversampling_010': ('nnUNetPlans',),
        'nnUNetTrainerFocalLoss2': ('nnUNetPlans',),
        'nnUNetTrainerFocalLoss': ('nnUNetPlans',),
        'nnUNetTrainerFocalLoss3': ('nnUNetPlans',),
        'nnUNetTrainerFocalandDiceLoss': ('nnUNetPlans',),
        'nnUNetTrainerAdan': ('nnUNetPlans',),
        'nnUNetTrainerAdan3en4': ('nnUNetPlans',),
        'nnUNetTrainerAdan1en3': ('nnUNetPlans',),
        'nnUNetTrainerAdam': ('nnUNetPlans',),
        'nnUNetTrainerAdam3en4': ('nnUNetPlans',),
        'nnUNetTrainerAdam1en3': ('nnUNetPlans',),
        'nnUNetTrainerAdanCosAnneal': ('nnUNetPlans',),
        'nnUNetTrainerCosAnneal': ('nnUNetPlans',),
        'nnUNetTrainerResEncUNet': ('nnUNetPlans',),
        'nnUNetTrainerCELoss': ('nnUNetPlans',),
        'nnUNetTrainerCELossLS01': ('nnUNetPlans',),
        'nnUNetTrainerTopk10Loss': ('nnUNetPlans',),
        'nnUNetTrainerTopk10LossLS01': ('nnUNetPlans',),
        'nnUNetTrainerDiceTopK10Loss': ('nnUNetPlans',),
        'nnUNetTrainerDiceLossClip1': ('nnUNetPlans',),
        'nnUNetTrainerDiceLoss': ('nnUNetPlans',),
        'nnUNetTrainerDiceLossLS01': ('nnUNetPlans',),
        'nnUNetTrainerDiceCELossLS01': ('nnUNetPlans',),
        'nnUNetTrainerNoMirroring': ('nnUNetPlans',),
        'nnUNetTrainerNoDA': ('nnUNetPlans',),
        'nnUNetTrainerNoDeepSupervision': ('nnUNetPlans',),
        'nnUNetTrainerDA5': ('nnUNetPlans',),
        'nnUNetTrainerDA5ord0': ('nnUNetPlans',),
        'nnUNetTrainerDiceCELossClip1': ('nnUNetPlans',),
        'nnUNetTrainerDAOrd0': ('nnUNetPlans',),
    }
    all_results_file= join(nnUNet_results, 'customDecResults.csv')
    datasets = [2, 3, 4, 17, 20, 24, 27, 38, 55, 64, 82]
    collect_results(use_these_trainers, datasets, all_results_file)

    folds = (0, 1, 2, 3, 4)
    configs = ("3d_fullres", "3d_lowres")
    output_file = join(nnUNet_results, 'customDecResults_summary5fold.csv')
    summarize(all_results_file, output_file, folds, configs, datasets, use_these_trainers)

    folds = (0, )
    configs = ("3d_fullres", "3d_lowres")
    output_file = join(nnUNet_results, 'customDecResults_summaryfold0.csv')
    summarize(all_results_file, output_file, folds, configs, datasets, use_these_trainers)

