import os
import shutil
import subprocess
from pathlib import Path

"""
To run these tests do
python tests/tests_nnunet.py
"""
def run_tests_and_exit_on_failure():

    # Set nnUNet_results env var
    weights_dir = Path.home() / "github_actions_nnunet" / "results"
    os.environ["nnUNet_results"] = str(weights_dir)
    print(f"Using weights directory: {weights_dir}")

    # Copy example file
    os.makedirs("tests/nnunet_input_files", exist_ok=True)
    shutil.copy(Path.home() / "github_actions_nnunet" / "example_ct_sm.nii.gz", "tests/nnunet_input_files/example_ct_sm_0000.nii.gz")

    # Run nnunet
    subprocess.call(f"nnUNetv2_predict -i tests/nnunet_input_files -o tests/nnunet_input_files -d 300 -tr nnUNetTrainer -c 3d_fullres -f 0 -device cpu", shell=True)

    # Check if output file exists
    assert os.path.exists("tests/nnunet_input_files/example_ct_sm.nii.gz"), "A nnunet output file was not generated."

    # Clean up
    shutil.rmtree("tests/nnunet_input_files")
    shutil.rmtree(Path.home() / "github_actions_nnunet")


if __name__ == "__main__":
    run_tests_and_exit_on_failure()