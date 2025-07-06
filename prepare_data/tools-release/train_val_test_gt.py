import os
import shutil
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--source_folder", type=str, help="source folder")
argparser.add_argument("--target_folder", type=str, help="target folder")
argparser.add_argument("--pic_folder", type=str, help="pic folder to find txt files")
argparser.add_argument("--zoom", type=str, help="zoom level")
args = argparser.parse_args()

source_folder_args = args.source_folder

source_folder_list = [
    source_folder_args + "color/use",
    source_folder_args + "category/use",
    source_folder_args + "line_type/use",
    source_folder_args + "line_num/use",
    source_folder_args + "function/use",
    source_folder_args + "bidirection/use",
    source_folder_args + "boundary/use",
]

target_folder_args = args.target_folder

target_folder_list = [
    os.path.join(target_folder_args, "color"),
    os.path.join(target_folder_args, "category"),
    os.path.join(target_folder_args, "line_type"),
    os.path.join(target_folder_args, "num"),
    os.path.join(target_folder_args, "attribute"),
    os.path.join(target_folder_args, "direction"),
    os.path.join(target_folder_args, "boundary"),
]


def read_filenames(file_path):
    with open(file_path, "r") as f:
        filenames = f.read().splitlines()
    return filenames


pic_folder = args.pic_folder
train_filenames = read_filenames(
    os.path.join(pic_folder, args.zoom + "train_filenames.txt")
)
val_filenames = read_filenames(
    os.path.join(pic_folder, args.zoom + "val_filenames.txt")
)
test_filenames = read_filenames(
    os.path.join(pic_folder, args.zoom + "test_filenames.txt")
)

for source_folder, target_folder in zip(source_folder_list, target_folder_list):

    for folder in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_folder, folder), exist_ok=True)

    print("Copying images...")

    for file_list, folder in [
        (train_filenames, "train"),
        (val_filenames, "val"),
        # (test_filenames, "test"),
    ]:
        for filename in file_list:
            filename = filename.replace(".png", "-GT.png")
            source_file_path = os.path.join(source_folder, filename)
            target_file_path = os.path.join(target_folder, folder, filename)
            shutil.copy(source_file_path, target_file_path)

    print("Images copied successfully!")

source_folder_direction = source_folder_args + "_angle_direction/use"
target_folder_direction = os.path.join(target_folder_args, "angle_direction")


for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(target_folder_direction, folder), exist_ok=True)

print("Copying images...")


for file_list, folder in [
    (train_filenames, "train"),
    (val_filenames, "val"),
    # (test_filenames, "test"),
]:
    for filename in file_list:
        filename = filename.replace(".png", "-GT-direction.png")
        source_file_path = os.path.join(source_folder_direction, filename)
        target_file_path = os.path.join(target_folder_direction, folder, filename)
        shutil.copy(source_file_path, target_file_path)

print("Images copied successfully!")
