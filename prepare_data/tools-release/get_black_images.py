import os
from PIL import Image
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Get black images")
    parser.add_argument("--folder_path", help="annotation directory")
    parser.add_argument("--txt_folder", help="output txt name")
    args = parser.parse_args()
    return args


args = parse_args()

folder_path = args.folder_path


black_images = []


for filename in os.listdir(folder_path):
    if filename.endswith(".png"):

        img = Image.open(os.path.join(folder_path, filename))

        img_array = np.array(img)

        if np.all(img_array == 0):

            black_images.append(filename.replace(".png", "").replace("-GT", ""))


with open((args.txt_folder + ".txt"), "w") as f:
    for item in black_images:
        f.write("%s\n" % item)


print(f"Number of black images: {len(black_images)}")


print(f"Number of all images: {len(os.listdir(folder_path))}")
