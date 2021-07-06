# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile-gpu
FROM nvidia/cuda:9.0-cudnn7-runtime

# Installs necessary dependencies.
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    wget \
    curl \
    python\
    make \
    g++ \
    git \
    pkg-config  \
    build-essential cmake libgtk-3-dev  \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev \
    sed && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /
RUN mkdir /opencv_build
WORKDIR /opencv_build
RUN git clone https://github.com/opencv/opencv.git 
RUN git clone https://github.com/opencv/opencv_contrib.git
WORKDIR /opencv_build/opencv/
RUN mkdir build
WORKDIR /opencv_build/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..
RUN make
RUN make install
WORKDIR /
RUN rm -rf opencv_build
RUN git clone https://github.com/AlexeyAB/darknet.git
WORKDIR /darknet
RUN sed -i 's/GPU=0/GPU=1/' ./Makefile && \
    sed -i 's/CUDNN=0/CUDNN=1/' ./Makefile && \
    sed -i 's/OPENCV=0/OPENCV=1/' ./Makefile 
RUN wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
WORKDIR /darknet
RUN make
RUN chmod +x ./darknet
# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    pip install setuptools && \
    rm get-pip.py
RUN pip install albumentations matplotlib numpy opencv-python pandas rich scikit-learn
WORKDIR /root

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
RUN mkdir /root/preprocessing
COPY ./preprocessing/ /root/preprocessing/
WORKDIR /root/preprocessing/
RUN rm -rf ./output
RUN python main.py -n "run_1" -cls ./data/classes.txt
COPY ./model/our-model-yolov4-tiny.cfg /darknet/cfg/our-model-yolov4-tiny.cfg
WORKDIR /darknet
# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["./darknet", "detector train /root/preprocessing/run_1/darknet.data ./cfg/our-model-yolov4-tiny.cfg yolov4-tiny.conv.29 -dont_show > /root/preprocessing/run_1/train.log"]
CMD ["gsutil" "cp /root/preprocessing/run_1/weights gs://aisscv/"]