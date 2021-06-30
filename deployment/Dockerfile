FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ENV CUB_PATH="/opt/cub"
ARG MAKEFLAGS=-j6
#
# YOLO
#
RUN apt-get update && apt-get install -y --fix-missing make g++ && apt install -y --fix-missing pkg-config
RUN git clone https://github.com/AlexeyAB/darknet.git
RUN sed -i 's/GPU=0/GPU=1/' ./darknet/Makefile
RUN sed -i 's/CUDNN=0/CUDNN=1/' ./darknet/Makefile
RUN sed -i 's/OPENCV=0/OPENCV=1/' ./darknet/Makefile
RUN sed -i 's/LIBSO=0/LIBSO=1/' ./darknet/Makefile
WORKDIR /darknet
RUN make
RUN wget https://pjreddie.com/media/files/yolov3-tiny.weights
#
# Jetcam
#
SHELL ["/bin/bash", "-c"]
WORKDIR /
RUN git clone https://github.com/NVIDIA-AI-IOT/jetcam
WORKDIR jetcam
RUN python3 setup.py install
#
# NodeJS for Jupyter Widgets
#
RUN wget https://nodejs.org/dist/v12.13.0/node-v12.13.0-linux-arm64.tar.xz \
    && tar -xJf node-v12.13.0-linux-arm64.tar.xz
WORKDIR node-v12.13.0-linux-arm64
RUN cp -R * /usr/local/
RUN node -v
#
# Jupyter
#
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
WORKDIR /
RUN mkdir data
CMD jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &> /var/log/jupyter.log