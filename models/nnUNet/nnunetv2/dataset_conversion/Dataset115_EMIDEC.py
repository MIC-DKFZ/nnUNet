import shutil
from pathlib import Path

from nnunetv2.dataset_conversion.Dataset027_ACDC import make_out_dirs
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def copy_files(src_data_dir: Path, src_test_dir: Path, train_dir: Path, labels_dir: Path, test_dir: Path):
    """Copy files from the EMIDEC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in src_data_dir.iterdir() if f.is_dir()])
    patients_test = sorted([f for f in src_test_dir.iterdir() if f.is_dir()])

    # Copy training files and corresponding labels.
    for patient in patients_train:
        train_file = patient / "Images" / f"{patient.name}.nii.gz"
        label_file = patient / "Contours" / f"{patient.name}.nii.gz"
        shutil.copy(train_file, train_dir / f"{train_file.stem.split('.')[0]}_0000.nii.gz")
        shutil.copy(label_file, labels_dir)

    # Copy test files.
    for patient in patients_test:
        test_file = patient / "Images" / f"{patient.name}.nii.gz"
        shutil.copy(test_file, test_dir / f"{test_file.stem.split('.')[0]}_0000.nii.gz")

    return len(patients_train)


def convert_emidec(src_data_dir: str, src_test_dir: str, dataset_id=27):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id, task_name="EMIDEC")
    num_training_cases = copy_files(Path(src_data_dir), Path(src_test_dir), train_dir, labels_dir, test_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={
            "background": 0,
            "cavity": 1,
            "normal_myocardium": 2,
            "myocardial_infarction": 3,
            "no_reflow": 4,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, help="The EMIDEC dataset directory.")
    parser.add_argument("-t", "--test_dir", type=str, help="The EMIDEC test set directory.")
    parser.add_argument(
        "-d", "--dataset_id", required=False, type=int, default=115, help="nnU-Net Dataset ID, default: 115"
    )
    args = parser.parse_args()
    print("Converting...")
    convert_emidec(args.input_dir, args.test_dir, args.dataset_id)
    print("Done!")
