import os
import zipfile
from typing import Tuple

from batchgenerators.utilities.file_and_folder_operations import isdir, join, isfile, subdirs
from nnunetv2.utilities.file_path_utilities import *
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_results


def export_pretrained_model(dataset_name_or_id: str, output_file: str,
                            configurations: Tuple[str] = ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
                            trainer: str = 'nnUNetTrainer',
                            plans_identifier: str = 'nnUNetPlans',
                            folds: Tuple[int, ...] = (0, 1, 2, 3, 4),
                            strict: bool = True,
                            save_checkpoints: Tuple[str, ...] = ('checkpoint_final.pth'),
                            export_crossval_predictions: bool = False) -> None:
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    with(zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED)) as zipf:
        for c in configurations:
            trainer_output_dir = get_output_folder(dataset_name, trainer, plans_identifier, c)

            if not isdir(trainer_output_dir):
                if strict:
                    raise RuntimeError("Task %s is missing the model %s" % (dataset_name_or_id, m))
                else:
                    continue

            expected_fold_folder = ["fold_%d" % i if i != 'all' else 'fold_all' for i in folds]
            assert all([isdir(join(trainer_output_dir, i)) for i in expected_fold_folder]), \
                f"not all requested folds are present; {dataset_name} {c}; requersted folds: {folds}"

            assert isfile(join(trainer_output_dir, "plans.json")), f"plans.json missing, {dataset_name} {c}"

            for fold_folder in expected_fold_folder:
                # debug.json, does not exist yet
                source_file = join(trainer_output_dir, fold_folder, "debug.json")
                if isfile(source_file):
                    zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

                # all requested checkpoints
                for chk in save_checkpoints:
                    source_file = join(trainer_output_dir, fold_folder, chk)
                    zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

                # progress.png
                source_file = join(trainer_output_dir, fold_folder, "progress.png")
                zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

                # if it exists, network architecture.png
                source_file = join(trainer_output_dir, fold_folder, "network_architecture.pdf")
                if isfile(source_file):
                    zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

                # validation folder with all predicted segmentations etc
                if export_crossval_predictions:
                    source_folder = join(trainer_output_dir, fold_folder, "validation")
                    zipf.write(source_folder, os.path.relpath(source_folder, nnUNet_results))
                # just the summary.json file from the validation
                else:
                    source_file = join(trainer_output_dir, fold_folder, "validation", "summary.json")
                    zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

            source_folder = join(trainer_output_dir, f'crossval_results_folds_{folds_tuple_to_string(folds)}')
            if isdir(source_folder):
                if export_crossval_predictions:
                    zipf.write(source_folder, os.path.relpath(source_folder, nnUNet_results))
                else:
                    source_files = [
                        join(trainer_output_dir, f'crossval_results_folds_{folds_tuple_to_string(folds)}', i) for i in
                        ['summary.json', 'postprocessing.pkl', 'postprocessing.json']
                    ]
                    for s in source_files:
                        zipf.write(s, os.path.relpath(s, nnUNet_results))
            # plans
            source_file = join(trainer_output_dir, "plans.json")
            zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

        ensemble_dir = join(nnUNet_results, dataset_name, 'ensembles', dataset_name_or_id)

        if not isdir(ensemble_dir):
            print("No ensemble directory found for task", dataset_name_or_id)
            return
        subd = subdirs(ensemble_dir, join=False)
        valid = []
        for s in subd:
            v = check_if_valid(s, configurations, (trainer, nnunet_trainer_cascade), (plans_identifier))
            if v:
                valid.append(s)
        for v in valid:
            zipf.write(join(ensemble_dir, v, 'postprocessing.json'),
                       os.path.relpath(join(ensemble_dir, v, 'postprocessing.json'),
                                       nnUNet_results))
