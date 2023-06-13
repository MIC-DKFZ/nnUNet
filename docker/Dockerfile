# FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.9-pt1.11-v1

# nnUNet_v2
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/DIAGNijmegen/nnUNet_v2 /home/user/nnunet && \
    git -C /home/user/nnunet checkout pathology && \
    chown -R user /home/user/nnunet && \
    pip3 install -e /home/user/nnunet

# Additional packages
RUN python3 -m pip install shapely==1.7.1
RUN python3 -m pip install git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main
RUN python3 -m pip install albumentations==1.2.1

ARG SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN python3 -m pip install git+https://github.com/DIAGNijmegen/nnUNet@feature/pathology-augmentations 

# Update pixman 
RUN : \ 
    && sudo apt-get update \
    && sudo apt-get install meson \ 
    && wget https://www.cairographics.org/releases/pixman-0.42.2.tar.gz \ 
    && tar -xf pixman-0.42.2.tar.gz \ 
    && cd pixman-0.42.2 \ 
    && mkdir build \ 
    && cd build \ 
    && meson --prefix=/usr --buildtype=release \ 
    && ninja \ 
    && sudo ninja install
