FROM nvidia/cuda:10.0-runtime-ubuntu18.04

RUN apt-get update -y && apt-get install -y \
 python3.6 \
 python3-pip \
 curl \
 git

RUN pip3 install numpy && \
  pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl

RUN git clone https://github.com/MIC-DKFZ/nnUNet.git
RUN cd nnUNet && pip3 install -r requirements.txt && pip3 install -e .

COPY predict.sh /
COPY rename_input.py /

ENV INPUTDIR "/opt/input/"
ENV OUTPUTDIR "/opt/output/"
ENV RESULTS_FOLDER "/opt/results/"

RUN mkdir ${RESULTS_FOLDER}

ENTRYPOINT [ "/predict.sh" ]
