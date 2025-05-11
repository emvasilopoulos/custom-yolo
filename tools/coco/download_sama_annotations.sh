#!/bin/bash

# Default destination path is current directory
DEST_PATH=${1:-.}

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_PATH"
mkdir -p "$DEST_PATH/train"
mkdir -p "$DEST_PATH/val"


# Download and extract validation data
echo "Downloading validation data to $DEST_PATH..."
curl https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-val.zip -o "$DEST_PATH/sama-coco-val.zip"
unzip "$DEST_PATH/sama-coco-val.zip" -d "$DEST_PATH/val/"

# Download and extract training data
echo "Downloading training data to $DEST_PATH..."
curl https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-train.zip -o "$DEST_PATH/sama-coco-train.zip"
unzip "$DEST_PATH/sama-coco-train.zip" -d "$DEST_PATH/train/"
