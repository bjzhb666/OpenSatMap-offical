#!/bin/bash
# run all scripts in one script
# Usage: ./run_all.sh

red_echo() {
    echo -e "\033[31m$1\033[0m"
}

# ORI_PIC_USE=picuse20trainvaltest
# SAVE_FOLDER=picuse20save # save folder, all subfolders will be created in this folder
# JSON_FILE=8.13/all/trainval20.json
# TARGET_FOLDER=${SAVE_FOLDER}/pic20gt                  # save the results before cutting train val test
# TARGET_FOLDER_AFTER_SPLIT=${SAVE_FOLDER}/pic20gtsplit # save the results after splitting train val test
# ZOOM=20
# TRAIN_TXT=picuse20trainvaltest/20train_filenames.txt
# VAL_TXT=picuse20trainvaltest/20val_filenames.txt

ORI_PIC_USE=19picsusetrainvaltest
SAVE_FOLDER=picuse19save
JSON_FILE=8.13/all/trainval19.json
TARGET_FOLDER=${SAVE_FOLDER}/pic19gt
TARGET_FOLDER_AFTER_SPLIT=${SAVE_FOLDER}/pic19gtsplit
ZOOM=19
TRAIN_TXT=19picsusetrainvaltest/19train_filenames.txt
VAL_TXT=19picsusetrainvaltest/19val_filenames.txt


if [ "$ZOOM" -eq 19 ]; then
    CLIP_SIZE=512
    IMAGE_SIZE=2048
elif [ "$ZOOM" -eq 20 ]; then
    CLIP_SIZE=1024
    IMAGE_SIZE=4096
fi

red_echo "start running all scripts"
red_echo ""

# 1. make the files in the desired folders
red_echo "1. make the files in the desired folders"
rm -rf allpic
rm -rf allpic-cut

mkdir -p allpic/use

PIC_USE_ABS=$(realpath "$ORI_PIC_USE")

for file in "$PIC_USE_ABS/train/"*; do
    ln -s "$file" allpic/use/
done


for file in "$PIC_USE_ABS/val/"*; do
    ln -s "$file" allpic/use/
done

cp "$PIC_USE_ABS"/*.txt allpic/

PIC_USE=allpic
red_echo ""
# 2. run genertae_GT_multiclass_only-multi.py to get the ground truth
red_echo "2. run genertae_GT_multiclass_only-multi.py to get the ground truth"
python tools-release/generate_GT_multiclass_only-multi.py --file_name $JSON_FILE \
    --target_folder $TARGET_FOLDER --source_folder $PIC_USE --image_size $IMAGE_SIZE
red_echo ""
# 3. run python generate_GT_direction_tag.py
red_echo "3. run python tools-release/generate_GT_direction_tag-multi.py"
python tools-release/generate_GT_direction_tag-multi.py --file_name $JSON_FILE \
    --target_folder $TARGET_FOLDER --source_folder $PIC_USE --image_size $IMAGE_SIZE
red_echo ""
# 4. generate train val test tag GT
red_echo "4. generate train val test tag GT"
python tools-release/train_val_test_tag.py --source_folder $TARGET_FOLDER-mask-tag/use --target_folder $TARGET_FOLDER_AFTER_SPLIT \
    --pic_folder ${PIC_USE} --zoom $ZOOM
red_echo ""

# 5. split other GT train val test
red_echo "5. split other GT train val test"
python tools-release/train_val_test_gt.py --source_folder $TARGET_FOLDER --target_folder $TARGET_FOLDER_AFTER_SPLIT \
    --pic_folder ${PIC_USE} --zoom $ZOOM

# 6. cut images and GT into small pics
red_echo "6. cut images and GT"
bash tools-release/run_cut_satellite.sh $TARGET_FOLDER_AFTER_SPLIT ${ORI_PIC_USE} $CLIP_SIZE
red_echo ""

# 7. check the existence of npy files
red_echo "7. check the existence of npy files"
bash tools-release/check_npy.sh $TARGET_FOLDER_AFTER_SPLIT-cut
red_echo ""

# 8. find all black images
red_echo "8. find all black images"
bash tools-release/run_black_images.sh $TARGET_FOLDER_AFTER_SPLIT-cut ${ORI_PIC_USE}-cut
red_echo ""

# 9. remove all black images
red_echo "9. remove all black images"
bash tools-release/remove_black_image.sh ${ORI_PIC_USE}-cut/txt $TARGET_FOLDER_AFTER_SPLIT-cut \
    ${ORI_PIC_USE}-cut
red_echo ""

# 10. move used things to a new path
red_echo "10. move used things to a new path"
rm -rf ${SAVE_FOLDER}/final
mkdir -p ${SAVE_FOLDER}/final
mv ${ORI_PIC_USE}-cut ${SAVE_FOLDER}/final
mv $TARGET_FOLDER_AFTER_SPLIT-cut ${SAVE_FOLDER}/final
red_echo "all scripts have been run"
