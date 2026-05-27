import argparse
from typing import Union


def _split_is_deep_ensemble(split: dict) -> bool:
    return bool(split.get("deep_ensemble", False))


def _remove_deep_ensemble_splits(splits: list[dict]) -> list[dict]:
    return [s for s in splits if not _split_is_deep_ensemble(s)]


def append_deep_ensemble_splits(splits: list[dict], case_identifiers: list[str], num_members: int,
                                overwrite_deep_ensemble_splits: bool = False) -> list[dict]:
    if num_members < 1:
        raise ValueError("num_members must be >= 1.")
    if not isinstance(splits, list):
        raise ValueError("splits must be a list of dictionaries.")
    if any(not isinstance(s, dict) for s in splits):
        raise ValueError("splits must be a list of dictionaries.")
    if len(case_identifiers) == 0:
        raise ValueError("case_identifiers must not be empty.")

    existing_deep_ensemble_splits = [s for s in splits if _split_is_deep_ensemble(s)]
    if len(existing_deep_ensemble_splits) > 0 and not overwrite_deep_ensemble_splits:
        raise RuntimeError("splits already contains deep ensemble splits. Use overwrite_deep_ensemble_splits=True "
                           "to replace them.")

    all_case_identifiers = sorted(case_identifiers)
    updated_splits = _remove_deep_ensemble_splits(splits) if overwrite_deep_ensemble_splits else list(splits)

    for i in range(num_members):
        updated_splits.append({
            "train": all_case_identifiers,
            "val": all_case_identifiers,
            "deep_ensemble": True,
            "deep_ensemble_member": i,
        })

    return updated_splits


def create_or_update_deep_ensemble_splits(splits: list[dict] | None, case_identifiers: list[str],
                                          num_members: int = 5, num_cv_folds: int = 5, seed: int = 12345,
                                          overwrite_deep_ensemble_splits: bool = False) -> list[dict]:
    all_case_identifiers = sorted(case_identifiers)
    if len(all_case_identifiers) == 0:
        raise ValueError("case_identifiers must not be empty.")

    if splits is None:
        if num_cv_folds < 2:
            raise ValueError("num_cv_folds must be >= 2 when creating cross-validation splits.")
        if num_cv_folds > len(all_case_identifiers):
            raise ValueError(f"Cannot create {num_cv_folds} cross-validation folds from "
                             f"{len(all_case_identifiers)} cases.")
        from nnunetv2.utilities.crossval_split import generate_crossval_split

        splits = generate_crossval_split(all_case_identifiers, seed=seed, n_splits=num_cv_folds)

    return append_deep_ensemble_splits(
        splits, all_case_identifiers, num_members, overwrite_deep_ensemble_splits=overwrite_deep_ensemble_splits)


def _get_preprocessed_dataset_folder(dataset_name_or_id: Union[int, str]) -> str:
    from batchgenerators.utilities.file_and_folder_operations import isdir, join

    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    if not isdir(preprocessed_dataset_folder):
        raise RuntimeError(f"Preprocessed dataset folder does not exist: {preprocessed_dataset_folder}. "
                           f"Run nnUNetv2_plan_and_preprocess first.")
    return preprocessed_dataset_folder


def _get_preprocessed_configuration_folder(dataset_name_or_id: Union[int, str], plans_identifier: str,
                                           configuration: str) -> tuple[str, str]:
    from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, join

    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    preprocessed_dataset_folder = _get_preprocessed_dataset_folder(dataset_name_or_id)
    plans_file = join(preprocessed_dataset_folder, plans_identifier + ".json")
    if not isfile(plans_file):
        raise RuntimeError(f"Plans file does not exist: {plans_file}")

    plans_manager = PlansManager(plans_file)
    configuration_manager = plans_manager.get_configuration(configuration)
    preprocessed_configuration_folder = join(preprocessed_dataset_folder, configuration_manager.data_identifier)
    if not isdir(preprocessed_configuration_folder):
        raise RuntimeError(f"Preprocessed data folder for configuration {configuration} of plans identifier "
                           f"{plans_identifier} does not exist: {preprocessed_configuration_folder}. "
                           f"Run preprocessing for this configuration first.")
    return preprocessed_dataset_folder, preprocessed_configuration_folder


def _get_case_identifiers(preprocessed_configuration_folder: str) -> list[str]:
    from batchgenerators.utilities.file_and_folder_operations import subfiles

    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    if len(subfiles(preprocessed_configuration_folder, join=False)) == 0:
        raise RuntimeError(f"No preprocessed cases found in {preprocessed_configuration_folder}.")
    dataset_class = infer_dataset_class(preprocessed_configuration_folder)
    case_identifiers = sorted(dataset_class.get_identifiers(preprocessed_configuration_folder))
    if len(case_identifiers) == 0:
        raise RuntimeError(f"No preprocessed cases found in {preprocessed_configuration_folder}.")
    return case_identifiers


def create_deep_ensemble_splits(dataset_name_or_id: Union[int, str], configuration: str,
                                plans_identifier: str = "nnUNetPlans", num_members: int = 5,
                                num_cv_folds: int = 5, seed: int = 12345,
                                overwrite_deep_ensemble_splits: bool = False) -> list[dict]:
    from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, save_json

    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    if num_members < 1:
        raise ValueError("num_members must be >= 1.")

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_dataset_folder, preprocessed_configuration_folder = _get_preprocessed_configuration_folder(
        dataset_name, plans_identifier, configuration)
    case_identifiers = _get_case_identifiers(preprocessed_configuration_folder)

    splits_file = join(preprocessed_dataset_folder, "splits_final.json")
    if isfile(splits_file):
        splits = load_json(splits_file)
        if not isinstance(splits, list):
            raise ValueError(f"Expected {splits_file} to contain a list of splits.")
    else:
        splits = None

    non_deep_ensemble_splits = _remove_deep_ensemble_splits(splits) if splits is not None else []
    updated_splits = create_or_update_deep_ensemble_splits(
        splits, case_identifiers, num_members, num_cv_folds, seed,
        overwrite_deep_ensemble_splits=overwrite_deep_ensemble_splits)
    if splits is None:
        non_deep_ensemble_splits = _remove_deep_ensemble_splits(updated_splits)
    appended_fold_indices = list(range(len(non_deep_ensemble_splits), len(updated_splits)))

    save_json(updated_splits, splits_file, sort_keys=False)

    print(f"Dataset: {dataset_name}")
    print(f"Existing non-deep-ensemble folds: {len(non_deep_ensemble_splits)}")
    print(f"Appended deep ensemble folds: {num_members}")
    print(f"Deep ensemble fold indices: {appended_fold_indices}")
    print(f"Wrote split file: {splits_file}")
    print("WARNING: Deep ensemble folds use all training cases. Their validation metrics are biased and must not be "
          "used for model selection or performance estimation.")

    return updated_splits


def create_deep_ensemble_splits_entry_point():
    parser = argparse.ArgumentParser(
        "Create or update nnU-Net splits_final.json with full-training-set deep ensemble folds.")
    parser.add_argument("dataset_name_or_id", type=str, help="Dataset name or ID.")
    parser.add_argument("configuration", type=str, help="nnU-Net configuration, for example 2d or 3d_fullres.")
    parser.add_argument("-p", "--plans_identifier", default="nnUNetPlans", required=False,
                        help="Plans identifier. Default: nnUNetPlans")
    parser.add_argument("--num_members", type=int, default=5, required=False,
                        help="Number of deep ensemble folds to append. Default: 5")
    parser.add_argument("--num_cv_folds", type=int, default=5, required=False,
                        help="Number of CV folds to create if splits_final.json does not exist. Default: 5")
    parser.add_argument("--seed", type=int, default=12345, required=False,
                        help="Seed for CV split creation if splits_final.json does not exist. Default: 12345")
    parser.add_argument("--overwrite_deep_ensemble_splits", action="store_true", required=False,
                        help="Remove existing splits marked as deep ensemble splits before appending new ones.")
    args = parser.parse_args()

    create_deep_ensemble_splits(args.dataset_name_or_id, args.configuration, args.plans_identifier,
                                args.num_members, args.num_cv_folds, args.seed,
                                args.overwrite_deep_ensemble_splits)


if __name__ == "__main__":
    create_deep_ensemble_splits_entry_point()
