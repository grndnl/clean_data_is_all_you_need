# ::1ST STAGE - DEPENDENCIES::
FROM continuumio/miniconda3:23.5.2-0 AS build_depend

RUN apt-get update \
    && apt-get install -y build-essential libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

#######################################
## Make layoutlmv3 environment
#######################################

RUN mkdir /home/env_support/
WORKDIR /home/env_support/

COPY ./resources /home/env_support

RUN conda create --name layoutlmv3 python=3.7

# Set shell to use the layoutlmv3 env that was just created
## This is the same as doing a `conda activate layoutlmv3`
## From this point all pip or conda commands will be for the layoutlmv3 env
SHELL ["conda", "run", "--no-capture-output", "-n", "layoutlmv3", "/bin/bash", "-c"]

RUN conda update --all

RUN pip install -r requirements.txt

WORKDIR /home/env_support/unilm/layoutlmv3
RUN pip install -r requirements.txt

RUN pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
RUN pip install -e .

# Make it so that the layoutlmv3 env is active when you run the container
RUN echo "source activate layoutlmv3" > ~/.bashrc

#######################################
## Extra Additions
#######################################

# Setting the timezone
RUN apt-get update && apt-get install -y tzdata
ENV TZ="US/Pacific"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update &&\
    apt-get install poppler-utils -y &&\
    pip install PyMuPDF &&\
    pip install pdf2image &&\
    pip install PyPDF2 &&\
    pip install pdfminer

RUN pip install nb_black

RUN pip install click==8.0.4

RUN pip install scikit-image