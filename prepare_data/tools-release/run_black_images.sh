# CATEGORY_FOLDER='pic20gt-cut/category'
# txt_folder='pics20trainvaltest-cut/txt'
CATEGORY_FOLDER=$1/category
txt_folder="$2/txt"

mkdir -p "$txt_folder"

find "$CATEGORY_FOLDER"/* -maxdepth 0 -type d | while read -r folder; do 
   

    # echo "***not find subfolder***"
    echo "$folder"
    echo "$txt_folder/$(basename "$folder")"
    # echo "$txt_folder/$(basename "$folder")"
    # echo ""
    python tools-release/get_black_images.py --folder_path "$folder" --txt_folder "$txt_folder/$(basename "$folder")"

done