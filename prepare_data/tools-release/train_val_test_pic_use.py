"""
Cut the pictures in the source folder into training set, validation set and test set, 
and save the names of the pictures in the corresponding set to the file.
ratio: 6:2:2
"""

import os
import random
import shutil
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--zoom", type=str, help="zoom level")
argparser.add_argument("--source_folder", type=str, help="source folder")
argparser.add_argument("--target_folder", type=str, help="target folder")
args = argparser.parse_args()

ZOOM = args.zoom

random.seed(42)

source_folder = args.source_folder
target_folder = args.target_folder

train_folder = os.path.join(target_folder, "train")
val_folder = os.path.join(target_folder, "val")
test_folder = os.path.join(target_folder, "test")


for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)


png_files = [file for file in os.listdir(source_folder) if file.endswith(".png")]


random.shuffle(png_files)


train_split = int(0.6 * len(png_files))
val_split = int(0.8 * len(png_files))


train_files = png_files[:train_split]
val_files = png_files[train_split:val_split]
test_files = png_files[val_split:]


train_filenames = []
for file in train_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))
    train_filenames.append(file)


val_filenames = []
for file in val_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(val_folder, file))
    val_filenames.append(file)

test_filenames = []
for file in test_files:
    shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))
    test_filenames.append(file)

with open(os.path.join(target_folder, ZOOM + "train_filenames.txt"), "w") as f:
    f.write("\n".join(train_filenames))

with open(os.path.join(target_folder, ZOOM + "val_filenames.txt"), "w") as f:
    f.write("\n".join(val_filenames))

with open(os.path.join(target_folder, ZOOM + "test_filenames.txt"), "w") as f:
    f.write("\n".join(test_filenames))
