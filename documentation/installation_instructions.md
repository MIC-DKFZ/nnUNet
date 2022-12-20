# System requirements
nnU-Net has been tested on Linux (Ubuntu 18, 20, 22; centOS, RHEL). We do not provide support for other operating
systems. Windows support is planned for the future.

**Inference (making predictions on new data)** nnU-Net for now requires a GPU to do this. In the future, we will enable 
CPU support for predictions as well (this will be slow though!)! For inference the GPU should have 4 GB of VRAM and be 
relatively recent (RTX 20 series or newer)

For **training** a GPU with at least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080 or RTX 3090) is 
required. RTX 4090 not tested yet. Here we also recommend a strong CPU to go along with the GPU. At least 6 CPU cores 
(12 threads) are recommended. CPU requirements are mostly related to data augmentation and scale with the number of 
input channels and target structures. The faster the GPU, the better the CPU should be!

Example workstation configurations for training:
- CPU: Ryzen 5800X - 5900X or 7900X would be even better! We have not yet tested Intel Alder/Raptor lake but they will likely work as well :-)
- GPU: RTX 3090 24 GB
- RAM: 32 GB (64GB preferred)

Example Server configuration for training:
- CPU: 2x AMD EPYC7763 for a total of 128C/256T. 16C/GPU are highly recommended for the GPU model here!
- GPU: 8xA100 PCIe (price/performance superior to SXM variant + they use less power)
- RAM: 1 TB

(nnU-net by default uses one GPU per training. The server configuration can run 8 model trainings simultaneously)

Note that you will need to manually set the number of processes nnU-Net uses for data augmentation 

# Installation instructions
We strongly recommend you install nnU-Net in a virtual environment! Pip or anaconda are both fine! If you choose to 
compile pytorch from source (see below), you will need to use conda instead of pip. 

Use a recent version of Python! 3.9 or newer is guaranteed to work!

1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version and (IMPORTANT!) choose 
the highest CUDA version compatible with your drivers for maximum performance. 
**DO NOT JUST `PIP INSTALL NNUNET` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**. For maximum performance, consider 
[compiling pytorch yourself](https://github.com/pytorch/pytorch#from-source) (experienced users only!). 
2) Install nnU-Net depending on your use case:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running 
     **inference with pretrained models**:

       ```pip install nnunetv2```

    2) For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
3) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to
   set a few of environment variables. Please follow the instructions [here](documentation/setting_up_paths.md).
5) export OMP_NUM_THREADS=1
5) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate
   plots of the network topologies it generates (see [Model training](#model-training)). To install hiddenlayer,
   run the following commands:
    ```bash
    pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNet_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this
environment must be activated when executing the commands.

All nnU-Net commands have a `-h` option which gives information on how to use them.

A typical installation of nnU-Net can be completed in less than 5 minutes. If pytorch needs to be compiled from source
(which is what we currently recommend when using Turing GPUs), this can extend to more than an hour.
