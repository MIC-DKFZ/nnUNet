
import os
import subprocess


# ======== PREPROCESSING ========
env_vars = os.environ.copy()
env_vars['nnUNet_raw'] = "/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_raw"
env_vars['nnUNet_preprocessed'] = "/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_preprocessed"
env_vars['nnUNet_results'] = "/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_results"

process = subprocess.Popen([
    "uv", "run", "--extra", "cu124",
    "nnUNetv2_plan_and_preprocess",
    "-pl", "nnUNetPlannerResEncM", 
    "-d", "1",
    "-c", "3d_fullres",
    "-npfp", "8",
    "--verify_dataset_integrity"
], env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Stream STDOUT and STDERR after process completes
stdout, stderr = process.communicate()

for line in stdout.splitlines():
    print("STDOUT:", line)

for line in stderr.splitlines():
    print("STDERR:", line)

# ========= TRAINING ========
process = subprocess.Popen([
    "uv", "run", "--extra", "cu124",
    "nnUNetv2_train",
    "-d", "1",
    "-c", "3d_fullres",
    "-p", "nnUNetPlannerResEncM",
    "-t", "Task001_BrainTumorSegmentation",
    "-f", "0"
], env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)