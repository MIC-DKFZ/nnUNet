#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 02:00:00
#SBATCH --cpus-per-task=4
#SBATCH -A nsingla
## Use an account that has GPGPU access

module purge
module load python/3.10.8
source ~/venvs/venv-3.8.6/bin/activate

python3 dicom2tiff.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s