import shutil
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


if __name__ == "__main__":
    # MAMA-MIA dataset configuration
    mamamia_data_dir = "/mnt/cnet/PinkCC/dataset/MAMA-MIA"
    images_dir = join(mamamia_data_dir, "images")
    segmentations_dir = join(mamamia_data_dir, "segmentations", "expert")
    csv_path = "/Users/az-r-ow/Developer/PinkCC/dataset/train_test_splits.csv"

    task_id = 974
    task_name = "MAMAMIA"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # Setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    # Read train/test splits from CSV
    df = pd.read_csv(csv_path)
    train_cases = df["train_split"].dropna().tolist()
    test_cases = df["test_split"].dropna().tolist()

    print(f"Found {len(train_cases)} training cases and {len(test_cases)} test cases")

    # Process training cases
    for case in train_cases:
        case = case.strip()  # Remove any whitespace
        # Copy all 4 modality files
        for modality_idx in range(4):
            src_file = join(images_dir, case, f"{case}_{modality_idx:04d}.nii.gz")
            dst_file = join(imagestr, f"{case}_{modality_idx:04d}.nii.gz")

            if not isfile(src_file):
                print(f"Warning: Missing file {src_file}")
                continue

            shutil.copy(src_file, dst_file)

        # Copy segmentation
        src_seg = join(segmentations_dir, f"{case}.nii.gz")
        dst_seg = join(labelstr, f"{case}.nii.gz")

        if not isfile(src_seg):
            print(f"Warning: Missing segmentation {src_seg}")
            continue

        shutil.copy(src_seg, dst_seg)

    # Process test cases
    for case in test_cases:
        case = case.strip()  # Remove any whitespace
        # Copy all 4 modality files
        for modality_idx in range(4):
            src_file = join(images_dir, case, f"{case}_{modality_idx:04d}.nii.gz")
            dst_file = join(imagests, f"{case}_{modality_idx:04d}.nii.gz")

            if not isfile(src_file):
                print(f"Warning: Missing file {src_file}")
                continue

            shutil.copy(src_file, dst_file)

        # Copy segmentation
        src_seg = join(segmentations_dir, f"{case}.nii.gz")
        dst_seg = join(labelsts, f"{case}.nii.gz")

        if not isfile(src_seg):
            print(f"Warning: Missing segmentation {src_seg}")
            continue

        shutil.copy(src_seg, dst_seg)

    # Generate dataset.json
    generate_dataset_json(
        out_base,
        channel_names={
            0: "DCE_pre_contrast",
            1: "DCE_post_contrast_1",
            2: "DCE_post_contrast_2",
            3: "DCE_post_contrast_3",
        },
        labels={
            "background": 0,
            "tumor": 1,
        },
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        dataset_release="1.0",
        reference="MAMA-MIA challenge dataset",
        license="See MAMA-MIA challenge terms",
    )

    print(f"Dataset conversion complete!")
    print(f"Output directory: {out_base}")
    print(f"Training cases: {len(train_cases)}")
    print(f"Test cases: {len(test_cases)}")
