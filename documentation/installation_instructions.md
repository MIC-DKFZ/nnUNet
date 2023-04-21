# System requirements

## Operating System
nnU-Net has been tested on Linux (Ubuntu 18.04, 20.04, 22.04; centOS, RHEL), Windows and MacOS! It should work out of the box!

## Hardware requirements
We support GPU (recommended), CPU and Apple M1/M2 as devices (currently Apple mps does not implement 3D 
convolutions, so you might have to use the CPU on those devices).

### Hardware requirements for Training
We recommend you use a GPU for training as this will take a really long time on CPU or MPS (Apple M1/M2). 
For training a GPU with at least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080/3090 or RTX 4080/4090) is 
required. We also recommend a strong CPU to go along with the GPU. 6 cores (12 threads) 
are the bare minimum! CPU requirements are mostly related to data augmentation and scale with the number of 
input channels and target structures. Plus, the faster the GPU, the better the CPU should be!

### Hardware Requirements for inference
Again we recommend a GPU to make predictions as this will be substantially faster than the other options. However, 
inference times are typically still manageable on CPU and MPS (Apple M1/M2). If using a GPU, it should have at least 
4 GB of available (unused) VRAM.

### Example hardware configurations
Example workstation configurations for training:
- CPU: Ryzen 5800X - 5900X or 7900X would be even better! We have not yet tested Intel Alder/Raptor lake but they will likely work as well.
- GPU: RTX 3090 or RTX 4090
- RAM: 64GB
- Storage: SSD (M.2 PCIe Gen 3 or better!)

Example Server configuration for training:
- CPU: 2x AMD EPYC7763 for a total of 128C/256T. 16C/GPU are highly recommended for fast GPUs such as the A100!
- GPU: 8xA100 PCIe (price/performance superior to SXM variant + they use less power)
- RAM: 1 TB
- Storage: local SSD storage (PCIe Gen 3 or better) or ultra fast network storage

(nnU-net by default uses one GPU per training. The server configuration can run up to 8 model trainings simultaneously)

### Setting the correct number of Workers for data augmentation (training only)
Note that you will need to manually set the number of processes nnU-Net uses for data augmentation according to your 
CPU/GPU ratio. For the server above (256 threads for 8 GPUs), a good value would be 24-30. You can do this by 
setting the `nnUNet_n_proc_DA` environment variable (`export nnUNet_n_proc_DA=XX`). 
Recommended values (assuming a recent CPU with good IPC) are 10-12 for RTX 2080 ti, 12 for a RTX 3090, 16-18 for 
RTX 4090, 28-32 for A100. Optimal values may vary depending on the number of input channels/modalities and number of classes.

# Installation instructions
We strongly recommend that you install nnU-Net in a virtual environment! Pip or anaconda are both fine. If you choose to 
compile PyTorch from source (see below), you will need to use conda instead of pip. 

Use a recent version of Python! 3.9 or newer is guaranteed to work!

**nnU-Net v2 can coexist with nnU-Net v1! Both can be installed at the same time.**

1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version with support for your hardware (cuda, mps, cpu).
**DO NOT JUST `pip install nnunetv2` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**. For maximum speed, consider 
[compiling pytorch yourself](https://github.com/pytorch/pytorch#from-source) (experienced users only!). 
2) Install nnU-Net depending on your use case:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running 
     **inference with pretrained models**:

       ```pip install nnunetv2```

    2) For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you
   can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
3) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to
   set a few environment variables. Please follow the instructions [here](setting_up_paths.md).
4) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate
   plots of the network topologies it generates (see [Model training](how_to_use_nnunet.md#model-training)). 
To install hiddenlayer,
   run the following command:
    ```bash
    pip install --upgrade git+https://github.com/Mohinta2892/hiddenlayer.git
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNetv2_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this
environment must be activated when executing the commands. You can see what scripts/functions are executed by 
checking the entry_points in the setup.py file.

All nnU-Net commands have a `-h` option which gives information on how to use them.
