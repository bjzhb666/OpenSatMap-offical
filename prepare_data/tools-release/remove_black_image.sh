#!/bin/bash

# text_file_path='pics20trainvaltest-cut/txt/train.txt'
# GT_SOURCE_FOLDER='pic20gt-cut'
text_file_path=$1/train.txt
GT_SOURCE_FOLDER=$2
PIC_SOURCE_FOLDER=$3

# remove black images in GT
find "$GT_SOURCE_FOLDER"/* -maxdepth 0 -type d | while read -r folder; do 
    echo "$folder/train"
    echo "$folder/not_use_train"
    
    python tools-release/remove_black.py --txt_file_path "$text_file_path" \
        --source_folder_path "$folder/train" --target_folder_path "$folder/not_use_train" 
done

echo ""

# remove black images in PIC
PIC_TRAIN_FOLDER="$PIC_SOURCE_FOLDER/train"
PIC_NOT_USE_TRAIN_FOLDER="$PIC_SOURCE_FOLDER/not_use_train"
echo "$PIC_TRAIN_FOLDER"
echo "$PIC_NOT_USE_TRAIN_FOLDER"
python tools-release/remove_black.py --txt_file_path "$text_file_path" \
    --source_folder_path "$PIC_TRAIN_FOLDER" --target_folder_path "$PIC_NOT_USE_TRAIN_FOLDER"