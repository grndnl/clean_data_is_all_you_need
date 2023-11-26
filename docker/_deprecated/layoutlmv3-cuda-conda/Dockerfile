# ::1ST STAGE - DEPENDENCIES::
FROM nvidia/cuda:11.1.1-base-ubuntu20.04 AS build_depend

# Install dependencies
RUN apt-get update \
    && apt-get install build-essential -y \
    && apt-get install wget -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

#**************************************
# Install miniconda
#**************************************

# Where we will install miniconda
ENV CONDA_DIR /opt/conda

# Download, rename and extract the miniconda script
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/miniconda.sh \
    && /bin/bash /home/miniconda.sh -b -p /opt/conda \
    && rm -rf /home/*

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Setup the .bashrc file so that base auto activates the base env
RUN echo "source activate base" > ~/.bashrc

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

RUN pip install -r requirements.txt

WORKDIR /home/env_support/unilm/layoutlmv3
RUN pip install -r requirements.txt

RUN pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
RUN pip install -e .

# Make it so that the layoutlmv3 env is active when you run the container
RUN echo "source activate layoutlmv3" > ~/.bashrc