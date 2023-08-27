FROM --platform=linux/amd64 nvidia/cuda:12.2.0-base-ubuntu22.04
ENV TZ=Australia/Sydney
#RUN cp /usr/share/zoneinfo/Australia/Sydney /etc/localtime
WORKDIR app


# install python and pip
ENV DEBIAN_FRONTEND=noninteractive 
#RUN apt-get install -y tzdata
RUN apt-get update && apt-get install -y software-properties-common && apt-get install -y python3-pip && apt-get install -y curl
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get install -y python3.10
#ENV PATH="${PATH}:/usr/bin/python3.10"


# Install poetry 
RUN curl -sSL https://install.python-poetry.org | python3 - 
#RUN poetry self add keyrings.google-artifactregistry-auth && poetry config virtualenvs.create false && poetry config installer.max-workers 10

# Install dependencies for imagecodecs 

#RUN apt-get -y update && apt-get install -y gcc
#RUN apt-get install -y python3.9-dev
#RUN apt-get install -y build-essential libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev libwebp-dev libbz2-dev libopenjp2-7-dev libjpeg-turbo8-dev libjxr-dev liblcms2-dev libcharls-dev libaec-dev libbrotli-dev libsnappy-dev libzopfli-dev libgif-dev libtiff-dev 
#RUN apt install -y libjxr0 libjbig0 libaec0 libsnappy1v5 libblosc1 libgif7 libwebp6 libtiff5 libzopfli1 libopenjp2-7 liblcms2-2 libbrotli1
#RUN apt-get install -y build-essential python3-dev cython3 python3-setuptools python3-pip python3-wheel python3-numpy python3-pytest python3-blosc python3-brotli python3-snappy python3-lz4 libz-dev libblosc-dev liblzma-dev liblz4-dev libzstd-dev libpng-dev libwebp-dev libbz2-dev libopenjp2-7-dev libjpeg-turbo8-dev libjxr-dev liblcms2-dev libcharls-dev libaec-dev libbrotli-dev libsnappy-dev libzopfli-dev libgif-dev libtiff-dev
#RUN pip3 install imagecodecs==2021.4.28


## Set env variables
RUN echo "export nnUNet_raw='/app/nnUNet_raw/'" >> ~/.bashrc
RUN echo "export nnUNet_preprocessed='/app/nnUNet_preprocessed/'" >> ~/.bashrc
RUN echo "export nnUNet_results='/app/nnUNet_results/'" >> ~/.bashrc
#RUN echo "export PATH='/root/.local/bin:$PATH'" >> ~/.bashrc


ENV PATH="${PATH}:/root/.local/bin"
#Only for Apple silicon mac
ENV USE_NNPACK=0 


## Copy files 
COPY nnUNet_preprocessed ./nnUNet_preprocessed/
COPY nnUNet_raw ./nnUNet_raw/
COPY nnUNet_results ./nnUNet_results/
COPY models ./models/
COPY util ./util/
COPY dicom2tiff.py .
COPY exploratory_data_analysis.py .
COPY main_change_loss.py .
COPY main.py .
COPY pyproject.toml .
#COPY requirement.txt .
COPY dummy.py .


#RUN pip3 install -r requirement.txt

WORKDIR /app/models/nnUNet/

RUN poetry install

WORKDIR /app

#RUN python3.9 -m  pip install  .

RUN  poetry install

#WORKDIR /app

CMD ["python3.10" ,"dummy.py"]
#CMD ["python3","dicom2tiff.py"]