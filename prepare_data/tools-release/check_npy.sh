#!/bin/bash


TARGET_FOLDER="$1/mask_tag"

if [ -n "$(find "$TARGET_FOLDER" -mindepth 1 -maxdepth 1 -type d -exec sh -c 'ls -1 "{}"/*.npy 2>/dev/null' \;)" ]; then
    echo ".npy files exist in subfolders"
    find "$TARGET_FOLDER" -mindepth 1 -maxdepth 1 -type d -exec sh -c 'ls -1 "{}"/*.npy 2>/dev/null' \;
else
    echo "No .npy files exist in the subfolders"
fi
