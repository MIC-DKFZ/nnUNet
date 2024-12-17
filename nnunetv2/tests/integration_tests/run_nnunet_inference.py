import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def dice_score(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def run_tests_and_exit_on_failure():
    """
    Runs inference of a simple nnU-Net for CT body segmentation on a small example CT image 
    and checks if the output is correct.
    """
    # Set nnUNet_results env var
    weights_dir = Path.home() / "github_actions_nnunet" / "results"
    os.environ["nnUNet_results"] = str(weights_dir)

    # Copy example file
    os.makedirs("nnunetv2/tests/github_actions_output", exist_ok=True)
    shutil.copy("nnunetv2/tests/example_data/example_ct_sm.nii.gz", "nnunetv2/tests/github_actions_output/example_ct_sm_0000.nii.gz")

    # Run nnunet
    subprocess.call(f"nnUNetv2_predict -i nnunetv2/tests/github_actions_output -o nnunetv2/tests/github_actions_output -d 300 -tr nnUNetTrainer -c 3d_fullres -f 0 -device cpu", shell=True)

    # Check if the nnunet segmentation is correct
    img_gt = nib.load(f"nnunetv2/tests/example_data/example_ct_sm_T300_output.nii.gz").get_fdata()
    img_pred = nib.load(f"nnunetv2/tests/github_actions_output/example_ct_sm.nii.gz").get_fdata()
    dice = dice_score(img_gt, img_pred)
    images_equal = dice > 0.99  # allow for a small difference in the segmentation, otherwise the test will fail often
    assert images_equal, f"The nnunet segmentation is not correct (dice: {dice:.5f})."

    # Clean up
    shutil.rmtree("nnunetv2/tests/github_actions_output")
    shutil.rmtree(Path.home() / "github_actions_nnunet")


if __name__ == "__main__":
    run_tests_and_exit_on_failure()