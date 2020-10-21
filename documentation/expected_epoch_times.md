# Introduction
Trainings can take some time. A well-running training setup is essential to get the most of nnU-Net. nnU-Net does not 
require any fancy hardware, just a well-balanced system. We recommend at least 32 GB of RAM, 6 CPU cores (12 threads), 
SSD storage (this can be SATA and does not have to be PCIe. DO NOT use an external SSD connected via USB!) and a 
2080 ti GPU. If your system has multiple GPUs, the 
other components need to scale linearly with  the number of GPUs.

# Benchmark Details
To ensure your system is running as intended, we provide some benchmark numbers against which you can compare. Here 
are the details about benchmarking:

- We benchmark **2d**, **3d_fullres** and a modified 3d_fullres that uses 3x the default batch size (called **3d_fullres_large** here) 
- The datasets **Task02_Heart**, **Task05_Prostate** and **Task03_Liver** of the Medical Segmentation Decathlon are used 
(they provide a good spectrum of dataset properties)
- we use the **nnUNetTrainerV2_5epochs** trainer. This will run only for 5 epochs and it will skip validation. 
From the 5 epochs, we select the fastest one as the epoch time. 
- We will also be running the **nnUNetTrainerV2_5epochs_dummyLoad** trainer on the **3d_fullres** config. This trainer does not use 
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
nnUNet_train 3d_fullres nnUNetTrainerV2_5epochs TASKID 0 -p nnUNetPlansv2.1_plans_bs3x # optional, only for GPUs with more than 16GB of VRAM
```

The time we are interested in is the epoch time. You can find it in the text output (stdout) or the log file 
located in your `RESULTS_FOLDER`. Note that the trainers used here run for 5 epochs. Select the fastest time from your 
output as your benchmark time.

# Results

The following table shows the results we are getting on our servers/workstations. You should be seeing similar numbers when you 
run the benchmark on your server/workstation. Note that fluctuations of a couple of seconds are normal!


|                                   | V100 32GB SXM3 (DGX2) 350W | V100 32GB SXM2 300W | V100 32GB PCIe 250W | Titan RTX 24GB 280W | RTX 2080 ti 11GB 250W | Titan Xp 12GB 250W |
|-----------------------------------|----------------------------|---------------------|---------------------|---------------------|-----------------------|-------------------|
| Task002_Heart 2d                  |            65.63           |        69.07        |        73.22        |        82.27        |         99.39         |       183.71      |
| Task003_Liver 2d                  |            71.80           |        73.44        |        78.63        |        86.11        |         103.89        |       187.30      |
| Task005_Prostate 2d               |            69.68           |        70.07        |        76.85        |        88.04        |         106.97        |       187.38      |
| Task002_Heart 3d_fullres          |           156.13           |        166.32       |        177.91       |        142.74       |         174.60        |       499.65      |
| Task003_Liver 3d_fullres          |           137.08           |        144.83       |        157.05       |        114.78       |         146.90        |       500.74      |
| Task005_Prostate 3d_fullres       |           119.82           |        126.20       |        135.72       |        106.01       |         135.08        |       463.21      |
| Task002_Heart 3d_fullres dummy    |           153.41           |        160.44       |        172.28       |        136.90       |         163.52        |       497.51      |
| Task003_Liver 3d_fullres dummy    |           135.63           |        139.76       |        147.33       |        110.61       |         146.37        |       495.55      |
| Task005_Prostate 3d_fullres dummy |           115.65           |        121.48       |        130.71       |        102.03       |         129.16        |       464.14      |
| Task002_Heart 3d_fullres large    |           317.63           |        338.79       |        349.91       |        371.94       |          OOM          |        OOM        |
| Task003_Liver 3d_fullres large    |           271.54           |        285.41       |        295.42       |        324.74       |          OOM          |        OOM        |
| Task005_Prostate 3d_fullres large |           280.30           |        296.37       |        304.16       |        289.22       |          OOM          |        OOM        |

# Troubleshooting
Your epoch times are substantially slower than ours? That's not good! This section will help you figure out what is 
wrong. Note that each system is unique and we cannot help you find bottlenecks beyond providing the information 
presented in this section!

## First step: Make sure you have the right software!
In order to get maximum performance, you need to have pytorch compiled cuDNN 8.0.2 or newer. The easiest way to get 
that is by using the [Nvidia pytorch Docker](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). 
If you cannot use docker, you will need to compile pytorch 
yourself. For that, first download and install cuDNN 8.0.2 or newer from the [Nvidia homepage](https://developer.nvidia.com/cudnn), then follow the 
[instructions on the pytorch website](https://github.com/pytorch/pytorch#from-source) on how to compile it.

cuDNN 8.0.2 or newer is essential to get good performance from Turing GPUs. Pytorch 1.6.0 only comes with 7.6.5 which will not 
give you tensor core acceleration for mixed precision for 3D networks. 
If you are using Volta GPUs (V100) getting cuDNN 8.0.2 is not required to get a speed boost with mixed precision 
training (but it will still increase your speed!). 
Pascal GPUs will not profit from mixed precision training and should behave the same with 8.0.2 and 7.6.5. 

Future releases of pytorch will be compiled with a more recent version of cuDNN. It is worth to check your
cuDNN version before you take any action. To do that, run 

```bash
python -c 'import torch;print(torch.backends.cudnn.version())'
```

If the output is `8002` or higher, then you are good to go. If not you may have to take action.

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
