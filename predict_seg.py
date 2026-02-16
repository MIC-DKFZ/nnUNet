import subprocess

# Define the prediction command
cmd = [
    "nnUNetv2_predict",
    "-i", "Code/nnUNet/data/output_COR",
    "-o", "Code/nnUNet/data/pred",
    "-d", "613",
    "-c", "3d_fullres_test",
    "-tr", "nnUNetTrainer_2epochs",
    "-p", "nnUNetPlans",
    "--save_probabilities"
]

# Run the command
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"Prediction failed with error: {e}")