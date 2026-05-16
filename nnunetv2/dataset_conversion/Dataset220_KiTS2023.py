from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import json
import random
from datetime import datetime
from tqdm import tqdm
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def convert_kits2023(
    kits_base_dir: str, nnunet_dataset_id: int = 220, eval_split: float = None
):
    task_name = "KiTS2023"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    cases = subdirs(kits_base_dir, prefix="case_", join=False)

    if eval_split is not None:
        # Stratified split by hundreds (0-99, 100-199, 200-299, 400-499, 500-599)
        random.seed(42)

        # Group cases by hundred
        strata = {}
        for case in cases:
            case_num = int(case.split("_")[1])
            stratum = case_num // 100
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(case)

        # Split each stratum
        train_cases = []
        eval_cases = []

        for stratum_cases in strata.values():
            random.shuffle(stratum_cases)
            split_idx = int(len(stratum_cases) * (1 - eval_split))
            train_cases.extend(stratum_cases[:split_idx])
            eval_cases.extend(stratum_cases[split_idx:])

        # Save split info
        split_info = {
            "train_cases": sorted(train_cases),
            "eval_cases": sorted(eval_cases),
            "num_train": len(train_cases),
            "num_eval": len(eval_cases),
            "eval_ratio": eval_split,
            "random_seed": 42,
            "source_dataset": kits_base_dir,
            "split_date": datetime.now().isoformat(),
        }

        with open(join(out_base, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)

        print(
            f"Split dataset: {len(train_cases)} training, {len(eval_cases)} evaluation cases"
        )
    else:
        train_cases = cases
        eval_cases = []

    # Copy training cases
    print("Copying training cases...")
    for case in tqdm(train_cases, desc="Training"):
        shutil.copy(
            join(kits_base_dir, case, "imaging.nii.gz"),
            join(imagestr, f"{case}_0000.nii.gz"),
        )
        shutil.copy(
            join(kits_base_dir, case, "segmentation.nii.gz"),
            join(labelstr, f"{case}.nii.gz"),
        )

    # Copy evaluation cases
    if eval_cases:
        print("Copying evaluation cases...")
        for case in tqdm(eval_cases, desc="Evaluation"):
            shutil.copy(
                join(kits_base_dir, case, "imaging.nii.gz"),
                join(imagests, f"{case}_0000.nii.gz"),
            )
            shutil.copy(
                join(kits_base_dir, case, "segmentation.nii.gz"),
                join(labelsts, f"{case}.nii.gz"),
            )

    generate_dataset_json(
        out_base,
        {0: "CT"},
        labels={"background": 0, "kidney": (1, 2, 3), "masses": (2, 3), "tumor": 2},
        regions_class_order=(1, 3, 2),
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        dataset_name=task_name,
        reference="none",
        release="0.1.3",
        overwrite_image_reader_writer="NibabelIOWithReorient",
        description="KiTS2023",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_folder",
        type=str,
        help="The downloaded and extracted KiTS2023 dataset (must have case_XXXXX subfolders)",
    )
    parser.add_argument(
        "-d",
        required=False,
        type=int,
        default=220,
        help="nnU-Net Dataset ID, default: 220",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=None,
        help="Evaluation split ratio (e.g., 0.18 for 18%%). If not provided, all cases go to training.",
    )
    args = parser.parse_args()
    amos_base = args.input_folder
    convert_kits2023(amos_base, args.d, args.eval_split)

    # /media/isensee/raw_data/raw_datasets/kits23/dataset
