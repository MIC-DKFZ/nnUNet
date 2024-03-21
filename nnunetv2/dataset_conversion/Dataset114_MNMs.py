import csv
import os
import random
from pathlib import Path

import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json

from nnunetv2.dataset_conversion.Dataset027_ACDC import make_out_dirs
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import nnunetv2.paths as paths


def read_csv(csv_file: str):
    patient_info = {}

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        patient_index = headers.index("External code")
        ed_index = headers.index("ED")
        es_index = headers.index("ES")
        vendor_index = headers.index("Vendor")

        for row in reader:
            patient_info[row[patient_index]] = {
                "ed": int(row[ed_index]),
                "es": int(row[es_index]),
                "vendor": row[vendor_index],
            }

    return patient_info


# ------------------------------------------------------------------------------
# Conversion to nnUNet format
# ------------------------------------------------------------------------------
def convert_mnms(src_data_folder: Path, csv_file_name: str, dataset_id: int):
    out_dir, out_train_dir, out_labels_dir, out_test_dir = make_out_dirs(dataset_id, task_name="MNMs")
    patients_train = [f for f in (src_data_folder / "Training" / "Labeled").iterdir() if f.is_dir()]
    patients_test = [f for f in (src_data_folder / "Testing").iterdir() if f.is_dir()]

    patient_info = read_csv(str(src_data_folder / csv_file_name))

    save_cardiac_phases(patients_train, patient_info, out_train_dir, out_labels_dir)
    save_cardiac_phases(patients_test, patient_info, out_test_dir)

    # There are non-orthonormal direction cosines in the test and validation data.
    # Not sure if the data should be fixed, or we should skip the problematic data.
    # patients_val = [f for f in (src_data_folder / "Validation").iterdir() if f.is_dir()]
    # save_cardiac_phases(patients_val, patient_info, out_train_dir, out_labels_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={"background": 0, "LVBP": 1, "LVM": 2, "RV": 3},
        file_ending=".nii.gz",
        num_training_cases=len(patients_train) * 2,  # 2 since we have ED and ES for each patient
    )


def save_cardiac_phases(
    patients: list[Path], patient_info: dict[str, dict[str, int]], out_dir: Path, labels_dir: Path = None
):
    for patient in patients:
        print(f"Processing patient: {patient.name}")

        image = nib.load(patient / f"{patient.name}_sa.nii.gz")
        ed_frame = patient_info[patient.name]["ed"]
        es_frame = patient_info[patient.name]["es"]

        save_extracted_nifti_slice(image, ed_frame=ed_frame, es_frame=es_frame, out_dir=out_dir, patient=patient)

        if labels_dir:
            label = nib.load(patient / f"{patient.name}_sa_gt.nii.gz")
            save_extracted_nifti_slice(label, ed_frame=ed_frame, es_frame=es_frame, out_dir=labels_dir, patient=patient)


def save_extracted_nifti_slice(image, ed_frame: int, es_frame: int, out_dir: Path, patient: Path):
    # Save only extracted diastole and systole slices from the 4D H x W x D x time volume.
    image_ed = nib.Nifti1Image(image.dataobj[..., ed_frame], image.affine)
    image_es = nib.Nifti1Image(image.dataobj[..., es_frame], image.affine)

    # Labels do not have modality identifiers. Labels always end with 'gt'.
    suffix = ".nii.gz" if image.get_filename().endswith("_gt.nii.gz") else "_0000.nii.gz"

    nib.save(image_ed, str(out_dir / f"{patient.name}_frame{ed_frame:02d}{suffix}"))
    nib.save(image_es, str(out_dir / f"{patient.name}_frame{es_frame:02d}{suffix}"))


# ------------------------------------------------------------------------------
# Create custom splits
# ------------------------------------------------------------------------------
def create_custom_splits(src_data_folder: Path, csv_file: str, dataset_id: int, num_val_patients: int = 25):
    existing_splits = os.path.join(paths.nnUNet_preprocessed, f"Dataset{dataset_id}_MNMs", "splits_final.json")
    splits = load_json(existing_splits)

    patients_train = [f.name for f in (src_data_folder / "Training" / "Labeled").iterdir() if f.is_dir()]
    # Filter out any patients not in the training set
    patient_info = {
        patient: data
        for patient, data in read_csv(str(src_data_folder / csv_file)).items()
        if patient in patients_train
    }

    # Get train and validation patients for both vendors
    patients_a = [patient for patient, patient_data in patient_info.items() if patient_data["vendor"] == "A"]
    patients_b = [patient for patient, patient_data in patient_info.items() if patient_data["vendor"] == "B"]
    train_a, val_a = get_vendor_split(patients_a, num_val_patients)
    train_b, val_b = get_vendor_split(patients_b, num_val_patients)

    # Build filenames from corresponding patient frames
    train_a = [f"{patient}_frame{patient_info[patient][frame]:02d}" for patient in train_a for frame in ["es", "ed"]]
    train_b = [f"{patient}_frame{patient_info[patient][frame]:02d}" for patient in train_b for frame in ["es", "ed"]]
    train_a_mix_1, train_a_mix_2 = train_a[: len(train_a) // 2], train_a[len(train_a) // 2 :]
    train_b_mix_1, train_b_mix_2 = train_b[: len(train_b) // 2], train_b[len(train_b) // 2 :]
    val_a = [f"{patient}_frame{patient_info[patient][frame]:02d}" for patient in val_a for frame in ["es", "ed"]]
    val_b = [f"{patient}_frame{patient_info[patient][frame]:02d}" for patient in val_b for frame in ["es", "ed"]]

    for train_set in [train_a, train_b, train_a_mix_1 + train_b_mix_1, train_a_mix_2 + train_b_mix_2]:
        # For each train set, we evaluate on A, B and (A + B) respectively
        # See table 3 from the original paper for more details.
        splits.append({"train": train_set, "val": val_a})
        splits.append({"train": train_set, "val": val_b})
        splits.append({"train": train_set, "val": val_a + val_b})

    save_json(splits, existing_splits)


def get_vendor_split(patients: list[str], num_val_patients: int):
    random.shuffle(patients)
    total_patients = len(patients)
    num_training_patients = total_patients - num_val_patients
    return patients[:num_training_patients], patients[num_training_patients:]


if __name__ == "__main__":
    import argparse

    class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(add_help=False, formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="MNMs conversion utility helper. This script can be used to convert MNMs data into the expected nnUNet "
        "format. It can also be used to create additional custom splits, for explicitly training on combinations "
        "of vendors A and B (see `--custom-splits`).\n"
        "If you wish to generate the custom splits, run the following pipeline:\n\n"
        "(1) Run `Dataset114_MNMs -i <raw_Data_dir>\n"
        "(2) Run `nnUNetv2_plan_and_preprocess -d 114 --verify_dataset_integrity`\n"
        "(3) Start training, but stop after initial splits are created: `nnUNetv2_train 114 2d 0`\n"
        "(4) Re-run `Dataset114_MNMs`, with `-s True`.\n"
        "(5) Re-run training.\n",
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default="./data/M&Ms/OpenDataset/",
        help="The downloaded MNMs dataset dir. Should contain a csv file, as well as Training, Validation and Testing "
        "folders.",
    )
    parser.add_argument(
        "-c",
        "--csv_file_name",
        type=str,
        default="211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
        help="The csv file containing the dataset information.",
    ),
    parser.add_argument("-d", "--dataset_id", type=int, default=114, help="nnUNet Dataset ID.")
    parser.add_argument(
        "-s",
        "--custom_splits",
        type=bool,
        default=False,
        help="Whether to append custom splits for training and testing on different vendors. If True, will create "
        "splits for training on patients from vendors A, B or a mix of A and B. Splits are tested on a hold-out "
        "validation sets of patients from A, B or A and B combined. See section 2.4 and table 3 from "
        "https://arxiv.org/abs/2011.07592 for more info.",
    )

    args = parser.parse_args()
    args.input_folder = Path(args.input_folder)

    if args.custom_splits:
        print("Appending custom splits...")
        create_custom_splits(args.input_folder, args.csv_file_name, args.dataset_id)
    else:
        print("Converting...")
        convert_mnms(args.input_folder, args.csv_file_name, args.dataset_id)

    print("Done!")
