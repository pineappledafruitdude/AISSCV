FROM docker.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL maintainer "Christopher Woelfle"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

ARG GPU=1
ARG CUDNN=1
ARG CUDNN_HALF=0
ARG OPENCV=1


# To save you a headache
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# Install needed libraries
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y clang-format wget apt-utils build-essential\
    checkinstall cmake pkg-config yasm git vim curl\
    autoconf automake libtool libopencv-dev build-essential


# Install python-dev
RUN apt-get update && apt-get install -y python3.8-dev python3-pip
# Install all needed python librarires 
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install albumentations matplotlib numpy opencv-python pandas rich scikit-learn

# Create working directory 
WORKDIR /

# Fetch Repo
RUN git clone https://github.com/AlexeyAB/darknet.git 
WORKDIR /darknet
RUN export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
# Modify Makefile according to needed env variables
RUN \ 
    sed -i "s/GPU=.*/GPU=${GPU}/" Makefile && \
    sed -i "s/CUDNN=0/CUDNN=${CUDNN}/" Makefile && \
    sed -i "s/CUDNN_HALF=0/CUDNN_HALF=${CUDNN_HALF}/" Makefile && \
    sed -i "s/OPENCV=0/OPENCV=${OPENCV}/" Makefile && \
    make
# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
WORKDIR /
RUN git clone https://github.com/pineappledafruitdude/AISSCV.git aisscv
WORKDIR /aisscv/model
RUN python3.8 create_cfg.py
WORKDIR /aisscv/preprocessing
RUN python3.8 main.py -n "run_1" -cls "./data/classes.txt" -o "/aisscv/model"

WORKDIR /darknet
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
# Sets up the entry point to invoke the trainer.
ENTRYPOINT ./darknet detector train /aisscv/model/run_1/darknet.data /aisscv/model/our-yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show && \
    rm -rf /aisscv/model/run_1/obj && \
    gsutil -m cp -r /aisscv/model/run_1/ gs://aisscv/$(date +%Y%m%d_%H%M%S)