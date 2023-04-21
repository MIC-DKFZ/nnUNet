import os
import shutil
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def make_out_dirs(dataset_id: int, task_name="ACDC"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_folder / "training").iterdir() if f.is_dir()])
    patients_test = sorted([f for f in (src_data_folder / "testing").iterdir() if f.is_dir()])

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                # The stem is 'patient.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.
                shutil.copy(file, train_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")
                num_training_cases += 1
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, labels_dir / file.name.replace("_gt", ""))

    # Copy test files.
    for patient_dir in patients_test:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                shutil.copy(file, test_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")

    return num_training_cases


def convert_acdc(src_data_folder: str, dataset_id=27):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "background",
        },
        labels={
            "background": 0,
            "RV": 1,
            "MLV": 2,
            "LVC": 3,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded ACDC dataset dir. Should contain extracted 'training' and 'testing' folders.",
    )
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=27, help="nnU-Net Dataset ID, default: 27"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_acdc(args.input_folder, args.dataset_id)
    print("Done!")
