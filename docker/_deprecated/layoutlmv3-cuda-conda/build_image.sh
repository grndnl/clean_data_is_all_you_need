#!/bin/bash


# Variables
DIVIDER_LINE="\n*******************************************************************************\n"
CURRENT_DIR=$PWD
IMAGE_NAME="layoutlmv3-cuda-conda:latest"

#--------------------------------------------------------------------------------------------------
printf $DIVIDER_LINE
printf "::BUILD IMAGE::\n\n"

docker build -t $IMAGE_NAME .