# ::1ST STAGE - DEPENDENCIES::
FROM continuumio/miniconda3:23.5.2-0 AS build_depend

RUN apt-get update \
    && apt-get install -y build-essential libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

#######################################
## Make conda-main environment
#######################################

RUN mkdir /home/env_support/
WORKDIR /home/env_support/

COPY ./resources /home/env_support

RUN conda create --name conda-main python=3.10

# Set shell to use the conda-main env that was just created
## This is the same as doing a `conda activate conda-main`
## From this point all pip or conda commands will be for the conda-main env
SHELL ["conda", "run", "--no-capture-output", "-n", "conda-main", "/bin/bash", "-c"]

RUN conda update --all
RUN conda install -y --file requirements_conda.txt
RUN pip install -r requirements_pip.txt

# Make it so that the conda-main env is active when you run the container
RUN echo "source activate conda-main" > ~/.bashrc

#######################################
## EXPERIMENTAL
#######################################
RUN apt-get update &&\
    apt-get install poppler-utils -y &&\
    pip install PyMuPDF &&\
    pip install pdf2image &&\
    pip install PyPDF2 &&\
    pip install pdfminer


# Setting the timezone
RUN apt-get update && apt-get install -y tzdata
ENV TZ="US/Pacific"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install wget