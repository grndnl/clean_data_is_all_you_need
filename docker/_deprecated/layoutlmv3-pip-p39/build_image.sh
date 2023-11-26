#!/bin/bash


# Variables
DIVIDER_LINE="\n*******************************************************************************\n"
CURRENT_DIR=$PWD
IMAGE_NAME="layoutlmv3-pip-p39:latest"

#--------------------------------------------------------------------------------------------------
printf $DIVIDER_LINE
printf "::BUILD IMAGE::\n\n"

docker build -t $IMAGE_NAME .