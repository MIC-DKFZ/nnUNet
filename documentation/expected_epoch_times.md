# Introduction
Trainings can take some time. A well-running training setup is essential to get the most of nnU-Net. nnU-Net does not 
require any fancy hardware, just a well-balanced system. We recommend at least 32 GB of RAM, 6 CPU cores (12 threads), 
SSD storage (this can be SATA and does not have to be PCIe. DO NOT use an external SSD connected via USB!) and a 
2080 ti GPU. If your system has multiple GPUs, the 
other components need to scale linearly with the number of GPUs.

# Benchmark Details
To ensure your system is running as intended, we provide some benchmark numbers against which you can compare. Here 
are the details about benchmarking:

- We benchmark **2d**, **3d_fullres** and a modified 3d_fullres that uses 3x the default batch size (called **3d_fullres large** here) 
- The datasets **Task002_Heart**, **Task005_Prostate** and **Task003_Liver** of the Medical Segmentation Decathlon are used 
(they provide a good spectrum of dataset properties)
- we use the nnUNetTrainerV2_5epochs trainer. This will run only for 5 epochs and it will skip validation. 
From the 5 epochs, we select the fastest one as the epoch time. 
- We will also be running the nnUNetTrainerV2_5epochs_dummyLoad trainer on the 3d_fullres config (called **3d_fullres dummy**). This trainer does not use 
the dataloader and instead uses random dummy inputs, bypassing all data augmentation (CPU) and I/O bottlenecks. 
- All trainings are done with mixed precision. This is why Pascal GPUs (Titan Xp) are so slow (they do not have 
tensor cores) 

# How to run the benchmark
First go into the folder where the preprocessed data and plans file of the task you would like to use are located. For me this is
`/home/fabian/data/nnUNet_preprocessed/Task002_Heart`

Then run the following python snippet. This will create our custom **3d_fullres_large** configuration. Note that this 
large configuration will only run on GPUs with 16GB or more! We included it in the test because some GPUs 
(V100, and probably also A100) can shine when they get more work to do per iteration.
```python
from batchgenerators.utilities.file_and_folder_operations import *
plans = load_pickle('nnUNetPlansv2.1_plans_3D.pkl')
stage = max(plans['plans_per_stage'].keys())
plans['plans_per_stage'][stage]['batch_size'] *= 3
save_pickle(plans, 'nnUNetPlansv2.1_bs3x_plans_3D.pkl')
```

Now you can run the benchmarks. Each should only take a couple of minutes
```bash
nnUNet_train 2d nnUNetTrainerV2_5epochs TASKID 0
nnUNet_train 3d_fullres nnUNetTrainerV2_5epochs TASKID 0
nnUNet_train 3d_fullres nnUNetTrainerV2_5epochs_dummyLoad TASKID 0
nnUNet_train 3d_fullres nnUNetTrainerV2_5epochs TASKID 0 -p nnUNetPlansv2.1_bs3x # optional, only for GPUs with more than 16GB of VRAM
```

The time we are interested in is the epoch time. You can find it in the text output (stdout) or the log file 
located in your `RESULTS_FOLDER`. Note that the trainers used here run for 5 epochs. Select the fastest time from your 
output as your benchmark time.

# Results

The following table shows the results we are getting on our servers/workstations. We are using pytorch 1.7.1 that we 
compiled ourselves using the instrucutions found [here](https://github.com/pytorch/pytorch#from-source). The cuDNN 
version we used is 8.1.0.77. You should be seeing similar numbers when you 
run the benchmark on your server/workstation. Note that fluctuations of a couple of seconds are normal!

IMPORTANT: Compiling pytorch from source is currently mandatory for best performance! Pytorch 1.8 does not have 
working tensorcore acceleration for 3D convolutions when installed with pip or conda!

IMPORTANT: A100 and V100 are very fast with the newer cuDNN versions and need more CPU workers to prevent bottlenecks,
set the environment variable `nnUNet_n_proc_DA=XX`
to increase the number of data augmentation workers. Recommended: 20 for V100, 32 for A100. Datasets with many input
modalities (BraTS: 4) require A LOT of CPU and should be used with even larger values for `nnUNet_n_proc_DA`

## Pytorch 1.7.1 compiled with cuDNN 8.1.0.77

|                                   | A100 40GB (DGX A100) 400W | V100 32GB SXM3 (DGX2) 350W | V100 32GB PCIe 250W | Quadro RTX6000 24GB 260W | Titan RTX 24GB 280W | RTX 2080 ti 11GB 250W | Titan Xp 12GB 250W |
|-----------------------------------|---------------------------|----------------------------|---------------------|--------------------------|---------------------|-----------------------|--------------------|
| Task002_Heart 2d                  | 40.06                     | 66.03                      | 76.19               | 78.01                    | 79.78               | 98.49                 | 177.87             |
| Task002_Heart 3d_fullres          | 51.17                     | 85.96                      | 99.29               | 110.47                   | 112.34              | 148.36                | 504.93             |
| Task002_Heart 3d_fullres dummy    | 48.53                     | 79                         | 89.66               | 105.16                   | 105.56              | 138.4                 | 501.64             |
| Task002_Heart 3d_fullres large    | 118.5                     | 220.45                     | 251.25              | 322.28                   | 300.96              | OOM                   | OOM                |
|                                   |                           |                            |                     |                          |                     |                       |                    |
| Task003_Liver 2d                  | 39.71                     | 60.69                      | 69.65               | 72.29                    | 76.17               | 92.54                 | 183.73             |
| Task003_Liver 3d_fullres          | 44.48                     | 75.53                      | 87.19               | 85.18                    | 86.17               | 106.76                | 290.87             |
| Task003_Liver 3d_fullres dummy    | 41.1                      | 70.96                      | 80.1                | 79.43                    | 79.43               | 101.54                | 289.03             |
| Task003_Liver 3d_fullres large    | 115.33                    | 213.27                     | 250.09              | 261.54                   | 266.66              | OOM                   | OOM                |
|                                   |                           |                            |                     |                          |                     |                       |                    |
| Task005_Prostate 2d               | 42.21                     | 68.88                      | 80.46               | 83.62                    | 81.59               | 102.81                | 183.68             |
| Task005_Prostate 3d_fullres       | 47.19                     | 76.33                      | 85.4                | 100                      | 102.05              | 132.82                | 415.45             |
| Task005_Prostate 3d_fullres dummy | 43.87                     | 70.58                      | 81.32               | 97.48                    | 98.99               | 124.73                | 410.12             |
| Task005_Prostate 3d_fullres large | 117.31                    | 209.12                     | 234.28              | 277.14                   | 284.35              | OOM                   | OOM                |

# Troubleshooting
Your epoch times are substantially slower than ours? That's not good! This section will help you figure out what is 
wrong. Note that each system is unique and we cannot help you find bottlenecks beyond providing the information 
presented in this section!

## First step: Make sure you have the right software!
In order to get maximum performance, you need to have pytorch compiled with a recent cuDNN version (8002 or newer is a must!). 
Unfortunately the currently provided pip/conda installable pytorch versions have a bug which causes their performance 
to be very low (see https://github.com/pytorch/pytorch/issues/57115 and https://github.com/pytorch/pytorch/issues/50153). 
They are about 2x-3x slower than the numbers we report in the table above. 
You need to have a pytorch version that was compiled from source to get maximum performance as shown in the table above.  
The easiest way to get that is by using the [Nvidia pytorch Docker](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). 
If you cannot use docker, you will need to compile pytorch 
yourself. For that, first download and install cuDNN from the [Nvidia homepage](https://developer.nvidia.com/cudnn), then follow the 
[instructions on the pytorch website](https://github.com/pytorch/pytorch#from-source) on how to compile it.

If you compiled pytorch yourself, you can check for the correct cuDNN version by running:
```bash
python -c 'import torch;print(torch.backends.cudnn.version())'
```
If the output is `8002` or higher, then you are good to go. If not you may have to take action. IMPORTANT: this 
only applies to pytorch that was compiled from source. pip/conda installed pytorch will report a new cuDNN version 
but still have poor performance due to the bug linked above.

## Identifying the bottleneck
If the software is up to date and you are still experiencing problems, this is how you can figure out what is going on:

While a training is running, run `htop` and `watch -n 0.1 nvidia-smi` (depending on your region you may have to use 
`0,1` instead). If you have physical access to the machine, also have a look at the LED indicating I/O activity.

Here is what you can read from that:
- `nvidia-smi` shows the GPU activity. `watch -n 0.1` makes this command refresh every 0.1s. This will allow you to 
see your GPU in action. A well running training will have your GPU pegged at 90-100% with no drops in GPU utilization. 
Your power should also be close to the maximum (for example `237W / 250 W`) at all times. 
- `htop` gives you an overview of the CPU usage. nnU-Net uses 12 processes for data augmentation + one main process. 
This means that up to 13 processes should be running simultaneously.
- the I/O LED indicates that your system is reading/writing data from/to your hard drive/SSD. Whenever this is 
blinking your system is doing something with your HDD/SSD.

### GPU bottleneck
If `nvidia-smi` is constantly showing 90-100% GPU utilization and the reported power draw is near the maximum, your 
GPU is the bottleneck. This is great! That means that your other components are not slowing it down. Your epochs times 
should be the same as ours reported above. If they are not then you need to investigate your software stack (see cuDNN stuff above).

What can you do about it?
1) There is nothing holding you back. Everything is fine!
2) If you need faster training, consider upgrading your GPU. Performance numbers are above, feel free to use them for guidance.
3) Think about whether you need more (slower) GPUs or less (faster) GPUs. Make sure to include Server/Workstation 
costs into your calculations. Sometimes it is better to go with more cheaper but slower GPUs run run multiple trainings 
in parallel.

### CPU bottleneck
You can recognize a CPU bottleneck as follows:
1) htop is consistently showing 10+ processes that are associated with your nnU-Net training
2) nvidia-smi is reporting jumps of GPU activity with zeroes in between

What can you do about it?
1) Depending on your single core performance, some datasets may require more than the default 12 processes for data 
augmentation. The CPU requirements for DA increase roughly linearly with the number of input modalities. Most datasets 
will train fine with much less than 12 (6 or even just 4). But datasets with for example 4 modalities may require more. 
If you have more than 12 CPU threads available, set the environment variable `nnUNet_n_proc_DA` to a number higher than 12.
2) If your CPU has less than 12 threads in total, running 12 threads can overburden it. Try lowering `nnUNet_n_proc_DA` 
to the number of threads you have available.
3) (sounds stupid, but this is the only other way) upgrade your CPU. I have seen Servers with 8 CPU cores (16 threads)
 and 8 GPUs in them. That is not well balanced. CPUs are cheap compared to GPUs. On a 'workstation' (single or dual GPU) 
 you can get something like a Ryzen 3900X or 3950X. On a server you could consider Xeon 6226R or 6258R on the Intel 
 side or the EPYC 7302P, 7402P, 7502P or 7702P on the AMD side. Make sure to scale the number of cores according to your 
 number of GPUs and use case. Feel free to also use our nnU-net recommendations from above.
 
### I/O bottleneck
On a workstation, I/O bottlenecks can be identified by looking at the LED indicating I/O activity. This is what an 
I/O bottleneck looks like:
- nvidia-smi is reporting jumps of GPU activity with zeroes in between
- htop is not showing many active CPU processes
- I/O LED is blinking rapidly or turned on constantly

Detecting I/O bottlenecks is difficult on servers where you may not have physical access. Tools like `iotop` are 
difficult to read and can only be run with sudo. However, the presence of an I/O LED is not strictly necessary. If
- nvidia-smi is reporting jumps of GPU activity with zeroes in between
- htop is not showing many active CPU processes

then the only possible issue to my knowledge is in fact an I/O bottleneck. 

Here is what you can do about an I/O bottleneck:
1) Make sure you are actually using an SSD to store the preprocessed data (`nnUNet_preprocessed`). Do not use an 
SSD connected via USB! Never use a HDD. Do not use a network drive that was not specifically designed to handle fast I/O 
(Note that you can use a network drive if it was designed for this purpose. At the DKFZ we use a
[flashblade](https://www.purestorage.com/products/file-and-object/flashblade.html) connected via ethernet and that works 
great)
2) A SATA SSD is only enough to feed 1-2 GPUs. If you have more GPUs installed you may have to upgrade to an nvme 
drive (make sure to get PCIe interface!).
