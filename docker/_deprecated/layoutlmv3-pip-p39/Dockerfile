# ::1ST STAGE - DEPENDENCIES::
FROM python:3.9.18-bullseye AS build_depend

RUN apt-get update &&\
    apt-get update -y &&\
    apt-get install -y build-essential libgl1-mesa-glx &&\
    rm -rf /var/lib/apt/lists/*

# GPU Access VARS
ENV NVIDIA_VISIBLE_DEVICES all

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Setting the timezone
RUN apt-get update && apt-get install -y tzdata
ENV TZ="US/Pacific"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

######################################
# Make layoutlmv3 environment
######################################

RUN mkdir /home/env_support/
WORKDIR /home/env_support/

COPY ./resources /home/env_support

RUN pip install --upgrade pip

RUN apt-get update &&\
    apt-get install poppler-utils -y &&\
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

WORKDIR /home/env_support/unilm/layoutlmv3
RUN pip install -r requirements.txt

RUN pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
RUN pip install -e .

######################################
# Extra Additions
######################################



