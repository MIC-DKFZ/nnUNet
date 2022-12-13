import zipfile

from nnunetv2.utilities.file_path_utilities import *


def export_pretrained_model(dataset_name_or_id: Union[int, str], output_file: str,
                            configurations: Tuple[str] = ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
                            trainer: str = 'nnUNetTrainer',
                            plans_identifier: str = 'nnUNetPlans',
                            folds: Tuple[int, ...] = (0, 1, 2, 3, 4),
                            strict: bool = True,
                            save_checkpoints: Tuple[str, ...] = ('checkpoint_final.pth',),
                            export_crossval_predictions: bool = False) -> None:
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    with(zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED)) as zipf:
        for c in configurations:
            print(f"Configuration {c}")
            trainer_output_dir = get_output_folder(dataset_name, trainer, plans_identifier, c)

            if not isdir(trainer_output_dir):
                if strict:
                    raise RuntimeError(f"{dataset_name} is missing the trained model of configuration {c}")
                else:
                    continue

            expected_fold_folder = ["fold_%d" % i if i != 'all' else 'fold_all' for i in folds]
            assert all([isdir(join(trainer_output_dir, i)) for i in expected_fold_folder]), \
                f"not all requested folds are present; {dataset_name} {c}; requested folds: {folds}"

            assert isfile(join(trainer_output_dir, "plans.json")), f"plans.json missing, {dataset_name} {c}"

            for fold_folder in expected_fold_folder:
                print(f"Exporting {fold_folder}")
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
                    files = [i for i in subfiles(source_folder, join=False) if not i.endswith('.npz') and not i.endswith('.pkl')]
                    for f in files:
                        zipf.write(join(source_folder, f), os.path.relpath(join(source_folder, f), nnUNet_results))
                # just the summary.json file from the validation
                else:
                    source_file = join(trainer_output_dir, fold_folder, "validation", "summary.json")
                    zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

            source_folder = join(trainer_output_dir, f'crossval_results_folds_{folds_tuple_to_string(folds)}')
            if isdir(source_folder):
                if export_crossval_predictions:
                    source_files = subfiles(source_folder, join=True)
                else:
                    source_files = [
                        join(trainer_output_dir, f'crossval_results_folds_{folds_tuple_to_string(folds)}', i) for i in
                        ['summary.json', 'postprocessing.pkl', 'postprocessing.json']
                    ]
                for s in source_files:
                    if isfile(s):
                        zipf.write(s, os.path.relpath(s, nnUNet_results))
            # plans
            source_file = join(trainer_output_dir, "plans.json")
            zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))
            # fingerprint
            source_file = join(trainer_output_dir, "dataset_fingerprint.json")
            zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))
            # dataset
            source_file = join(trainer_output_dir, "dataset.json")
            zipf.write(source_file, os.path.relpath(source_file, nnUNet_results))

        ensemble_dir = join(nnUNet_results, dataset_name, 'ensembles')

        if not isdir(ensemble_dir):
            print("No ensemble directory found for task", dataset_name_or_id)
            return
        subd = subdirs(ensemble_dir, join=False)
                # figure out whether the models in the ensemble are all within the exported models here
        for ens in subd:
            identifiers, folds = convert_ensemble_folder_to_model_identifiers_and_folds(ens)
            ok = True
            for i in identifiers:
                tr, pl, c = convert_identifier_to_trainer_plans_config(i)
                if tr == trainer and pl == plans_identifier and c in configurations:
                    pass
                else:
                    ok = False
            if ok:
                print(f'found matching ensemble: {ens}')
                source_folder = join(ensemble_dir, ens)
                if export_crossval_predictions:
                    source_files = subfiles(source_folder, join=True)
                else:
                    source_files = [
                        join(source_folder, i) for i in
                        ['summary.json', 'postprocessing.pkl', 'postprocessing.json'] if isfile(join(source_folder, i))
                    ]
                for s in source_files:
                    zipf.write(s, os.path.relpath(s, nnUNet_results))
    print('Done')


if __name__ == '__main__':
    export_pretrained_model(2, '/home/fabian/temp/dataset2.zip', strict=False, export_crossval_predictions=True, folds=(0, ))