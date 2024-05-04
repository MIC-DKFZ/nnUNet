FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

#WORKDIR /home/tmi/cardiologs/nnUNet
#COPY . .
# RUN pip install -e .

RUN pip install hiddenlayer
