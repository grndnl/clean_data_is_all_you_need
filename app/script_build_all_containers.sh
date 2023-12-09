#!/bin/bash


# Variables
DIVIDER_LINE="\n*******************************************************************************\n"
CURRENT_DIR=$PWD


#--------------------------------------------------------------------------------------------------
DLA_IMAGE_NAME="clean_dla:latest"
DLA_DIRECTORY=$CURRENT_DIR/dla

printf $DIVIDER_LINE
printf "::BUILDING $DLA_IMAGE_NAME::\n\n"

docker build -t $DLA_IMAGE_NAME $DLA_DIRECTORY

#--------------------------------------------------------------------------------------------------
# Commented since we are not using this image

# TEXT_IMAGE_NAME="clean_text_extraction:latest"
# TEXT_DIRECTORY=$CURRENT_DIR/text_extraction

# printf $DIVIDER_LINE
# printf "::BUILDING $TEXT_IMAGE_NAME::\n\n"

# docker build -t $TEXT_IMAGE_NAME $TEXT_DIRECTORY

