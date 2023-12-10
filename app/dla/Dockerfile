# ::1ST STAGE - DEPENDENCIES::
FROM nvidia/cuda:11.3.1-devel-ubuntu18.04 AS build_depend

ENV DEBIAN_FRONTEND noninteractive

# Define locales (necessary for the nvidia/cuda:11.3.1-devel-ubuntu18.04 image)
RUN apt-get update &&\
    apt-get install -y locales &&\
    localedef -i en_US -f UTF-8 en_US.UTF-8 &&\
    rm -rf /var/lib/apt/lists/*

# Install main libraries
RUN apt-get update &&\
    apt-get install -y tzdata wget curl git ninja-build python3-opencv poppler-utils &&\
    rm -rf /var/lib/apt/lists/*

# Setting the timezone
ENV TZ="US/Pacific"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Define env directory
ENV APP_DIR /app

ENV ENV_NAME venv
ENV ENV_DIR ${APP_DIR}/${ENV_NAME}

WORKDIR $APP_DIR

#######################################
## Install Miniconda
#######################################

# Where we will install miniconda
ENV CONDA_DIR /opt/conda

# Download, rename and extract the miniconda script
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_23.10.0-1-Linux-x86_64.sh -O /home/miniconda.sh \
    && /bin/bash /home/miniconda.sh -b -p /opt/conda \
    && rm -rf ./*

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Create App Environment
RUN conda create -p ${ENV_DIR} python=3.8 -y

# Put conda in path so it becomes the default
ENV PATH=${ENV_DIR}/bin:$PATH

# #######################################
# ## Install Torch and Detectron
# #######################################

ENV ENV_SUPPORT_DIR ${ENV_NAME}_support
WORKDIR ${ENV_SUPPORT_DIR}

# Install Torch
RUN pip install tensorboard cmake onnx
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install Detectron dependencies
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

# Download Detectron 2 and Set ENV
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install Detectron2
RUN pip install -e detectron2_repo

#######################################
## Install layoutlmv3 ENV Dependencies
#######################################

COPY ./docker_resources ./

RUN pip install -r requirements_layoutlmv3.txt
RUN pip install -r requirements.txt
RUN pip install -r requirements_web_api.txt

# #######################################
# ## Later Additions
# #######################################

RUN apt-get update &&\
    apt-get install -y tesseract-ocr &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_DIR}

COPY . ./