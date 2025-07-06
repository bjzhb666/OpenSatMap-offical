SOURCE_FOLDER=$1 # Folder where the train val test is stored
TARGET_FOLDER="$SOURCE_FOLDER-cut"
CLIP_SIZE=$3

# Get the names of all direct subfolders and output them in a loop
find "$SOURCE_FOLDER"/* -maxdepth 0 -type d | while read -r folder; do
    find "$folder"/* -maxdepth 0 -type d | while read -r subfolder; do
        echo "$subfolder"
        echo "$TARGET_FOLDER/$(basename "$folder")/$(basename "$subfolder")"
        python tools-release/cut_satellite.py -dataset_path "$subfolder" -o "$TARGET_FOLDER/$(basename "$folder")/$(basename "$subfolder")" \
            --is_GT --clip_size $CLIP_SIZE --stride_size $CLIP_SIZE
    done
    echo ""
done

echo "start cutting images"
echo ""

PIC_SOURCE=$2
PIC_TARGET="$PIC_SOURCE-cut"

find "$PIC_SOURCE"/* -maxdepth 0 -type d | while read -r folder; do
 
    echo "$folder"
    echo "$PIC_TARGET/$(basename "$folder")"
    python tools-release/cut_satellite.py -dataset_path "$folder" -o "$PIC_TARGET/$(basename "$folder")" \
        --clip_size  $CLIP_SIZE --stride_size  $CLIP_SIZE
done
