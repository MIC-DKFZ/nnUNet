# System requirements
nnU-Net has been tested on Linux (Ubuntu 18, 20, 22; centOS, RHEL). We do not provide support for other operating
systems. Windows support is planned for the future.

**Inference (making predictions on new data)** nnU-Net for now requires a GPU to do this. In the future, we will enable 
CPU support for predictions as well (this will be slow though!)! For inference the GPU should have 4 GB of VRAM and be 
relatively recent (RTX 20 series or newer)

For **training** a GPU with at least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080 or RTX 3090) is 
required. RTX 40** series not tested yet. We also recommend a strong CPU to go along with the GPU. 6 cores (12 threads) 
are the bare minimum! CPU requirements are mostly related to data augmentation and scale with the number of 
input channels and target structures. Plus, the faster the GPU, the better the CPU should be!

Example workstation configurations for training:
- CPU: Ryzen 5800X - 5900X or 7900X would be even better! We have not yet tested Intel Alder/Raptor lake but they will likely work as well.
- GPU: RTX 3090 24 GB
- RAM: 64GB
- Storage: SD (PCIe Gen 3 or better!0)

Example Server configuration for training:
- CPU: 2x AMD EPYC7763 for a total of 128C/256T. 16C/GPU are highly recommended for the GPU model here!
- GPU: 8xA100 PCIe (price/performance superior to SXM variant + they use less power)
- RAM: 1 TB
- Storage: local SSD storage (PCIe Gen 3 or better) or ultra fast network storage

(nnU-net by default uses one GPU per training. The server configuration can run 8 model trainings simultaneously)

Note that you will need to manually set the number of processes nnU-Net uses for data augmentation according to your 
CPU/GPU ratio. For the server above (256 threads for 8 GPUs), a good value would be 24-30. You can do this by 
setting the nnUNet_n_proc_DA environment variable (export nnUNet_n_proc_DA=XX). 

# Installation instructions
We strongly recommend you install nnU-Net in a virtual environment! Pip or anaconda are both fine! If you choose to 
compile pytorch from source (see below), you will need to use conda instead of pip. 

Use a recent version of Python! 3.9 or newer is guaranteed to work!

nnU-Net v2 can coexist with nnU-Net v1! Both can be installed at the same time.

1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version and (IMPORTANT!) choose 
the highest CUDA version compatible with your drivers for maximum performance (check cuda version in nvidia-smi on Linux). 
**DO NOT JUST `pip install nnunetv2` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**. For maximum performance, consider 
[compiling pytorch yourself](https://github.com/pytorch/pytorch#from-source) (experienced users only!). 
2) Install dependencies:
   ```bash
   pip install --upgrade git+https://github.com/MIC-DKFZ/acvl_utils.git
   pip install --upgrade git+https://github.com/MIC-DKFZ/dynamic-network-architectures.git
   ```
   (these will be provided as proper python packages in the future)
3) Install nnU-Net depending on your use case:
    1) [DOES NOT WORK YET] For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running 
     **inference with pretrained models**:

       ```pip install nnunetv2```

    2) [USE THIS!] For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          git checkout nnunet_remake
          git pull  # just for good measure
          pip install -e .
          ```
4) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to
   set a few of environment variables. Please follow the instructions [here](documentation/setting_up_paths.md).
5) Add OMP_NUM_THREADS=1 to your environment. Linux/bash example: add `export OMP_NUM_THREADS=1` to your .bashrc file. 
Alternatively you can always specify it when running nnU-Net: `OMP_NUM_THREADS=1 nnUNetv2_train [...]` 
6) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate
   plots of the network topologies it generates (see [Model training](#model-training)). To install hiddenlayer,
   run the following command:
    ```bash
    pip install --upgrade git+https://github.com/julien-blanchon/hiddenlayer.git
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNetv2_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this
environment must be activated when executing the commands. You can see what scripts/functions are executed by 
checking the entry_points in the setup.py file.

All nnU-Net commands have a `-h` option which gives information on how to use them.
