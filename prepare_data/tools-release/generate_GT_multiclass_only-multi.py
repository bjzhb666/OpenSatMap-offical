import cv2
import numpy as np
import json
import os
from PIL import Image
import argparse

LINE_WIDTH = 1
VISUALIZATION = False


def generate_line_mask_without_direction(coordinates, line_width, image_size):
    """
    generate the mask of the line without considering the direction
    :param coordinates:
    :param line_width:
    :param image_size:
    :return: mask of the line
    """
    if len(coordinates) < 2:
        return None

    mask = np.zeros(image_size, dtype=np.uint8)

    coordinates = np.array(coordinates, dtype=np.int32)

    cv2.polylines(mask, [coordinates], False, 255, thickness=line_width)

    indices = np.where(mask == 255)
    line_points = np.column_stack((indices[1], indices[0]))

    return line_points


def generate_pic_mask(coordinates, pixel_value=1, IMAGE_SIZE=4096):
    """
    Generate a mask for the entire image, one instance generates one mask
    :param coordinates: Coordinate points of the line
    :param pixel_value: Pixel value of the line
    :param too_long: If the number of tags exceeds 255 when generating tags, you need to create a full_mask with dtype of uint16
    :return: Image mask, size=IMAGE_SIZE,IMAGE_SIZE
    """
    image_size = (IMAGE_SIZE, IMAGE_SIZE)
    # Create a blank image as a complete mask
    full_mask = np.zeros(image_size, dtype=np.uint8)

    # Convert coordinate points to integers
    coordinates = np.array(coordinates, dtype=np.int32)
    # Delete points whose coordinates are outside the image range.
    coordinates = coordinates[coordinates[:, 0] < IMAGE_SIZE]
    coordinates = coordinates[coordinates[:, 1] < IMAGE_SIZE]
    # Set the specified coordinate point to pixel_value
    full_mask[coordinates[:, 1], coordinates[:, 0]] = pixel_value

    return full_mask


def read_json(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


line_cls2id = {
    "Lane line": 1,
    "Curb": 2,
    "Virtual line": 3,
}

color_cls2id = {
    "White": 1,
    "Yellow": 2,
    "Others": 3,
    "None": 4,
}

linetype_cls2id = {
    "Solid": 1,
    "Dashed": 2,
    "Thick solid": 3,
    "Short dashed": 4,
    "Others": 5,
    "None": 6,
}

linenum_cls2id = {
    "Single": 1,
    "Double": 2,
    "Others": 3,
    "None": 4,
}

# function
attribute_cls2id = {
    "None": 1,
    "No parking": 2,
    "Deceleration lane": 3,
    "Bus lane": 4,
    "Others": 5,
    "Tidal lane": 6,
    "Chevron markings": 7,
    "Parking space": 8,
    "Vechile staging area": 9,
    "Guidance line": 10,
    "Lane-borrowing area": 11,
}

direction_cls2id = {"False": 1, "True": 2}

boundary_cls2id = {"True": 1, "False": 2}


def generate_gt(
    image_name, target_path, json_data, IMAGE_SIZE, attr="category", idx=line_cls2id
):
    """
        generate the semantic segmentation gt label for each image
    Args:
        image_name (str):
        target_path (str):
        json_data (dict): annotation json data
        attr (str): the attribute of the line, including 'category', 'color', 'line_type', 'num', 'attribute', 'direction', 'boundary'
    Returns:
        None
    """

    png_name = image_name + ".png"
    pic_lines_list = json_data[png_name]["lines"]
    # print(pic_lines_list)
    len_lines = len(pic_lines_list)

    if len_lines == 0:
        # Returns an image of IMAGE_SIZE*IMAGE_SIZE with all zeros
        pic_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        cv2.imwrite(target_path, pic_mask)
        print("This image has no label")
        return

    # Create a flag map. If there is a line point that passes through two or more times,
    # then the coordinate point is invalid and it is marked as True.
    visited = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i in range(len_lines):
        line_key_points = pic_lines_list[i]["points"]
        line_points = generate_line_mask_without_direction(
            line_key_points, LINE_WIDTH, (IMAGE_SIZE, IMAGE_SIZE)
        )
        # Record the number of visits to each point
        if line_points is None:  # stupid annotation with the same point
            continue
        if len(line_points) == 0:  # this instance is invalid (no points)
            # import pdb; pdb.set_trace()
            continue

        for point in line_points:
            # import pdb; pdb.set_trace()
            x, y = point
            visited[y, x] += 1
            # Note that we put the column index first and the row index last.
            # This is because in image processing, columns (i.e. x coordinates) are usually specified first, then rows (i.e. y coordinates).
            # This is the opposite of the general matrix indexing (rows first, then columns).

    # Mark the points in visited that have a visit count greater than 1 as True
    visited = visited > 1

    # Start generating GT mask
    for i in range(len_lines):

        line_cls = pic_lines_list[i][attr]
        line_key_points = pic_lines_list[i]["points"]  # Points of an instance
        line_points = generate_line_mask_without_direction(
            line_key_points, LINE_WIDTH, (IMAGE_SIZE, IMAGE_SIZE)
        )

        if line_points is None:
            continue

        # generate the mask
        if i == 0:
            pic_mask = generate_pic_mask(
                line_points, pixel_value=idx[line_cls], IMAGE_SIZE=IMAGE_SIZE
            )
        else:
            pic_mask = pic_mask + generate_pic_mask(
                line_points, pixel_value=idx[line_cls], IMAGE_SIZE=IMAGE_SIZE
            )

    # Remove the points in visited that have a visit count greater than 1
    pic_mask[visited] = 255  # 255 is ignore value
    
    cv2.imwrite(target_path, pic_mask)

def visualize_seg_map(img):
    """
    Visualizes a segmentation map.
    Args:
        img (np.ndarray): The segmentation map to visualize.
    Returns:
        PIL.Image: Visualized segmentation map.
    """

    ignore_value = 255

    # Define the colors for all labels except the ignore value.
    color_map = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

    # Create an empty colored image
    colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Color the image
    for k in color_map.keys():
        colored_img[img == k] = color_map[k]

    # Set the ignore value regions to white
    colored_img[img == ignore_value] = [255, 255, 255]

    # Convert colored numpy array back to PIL Image
    img_pil = Image.fromarray(colored_img)

    return img_pil


import os
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Value, Lock

# Initialize the counter and the lock
counter = Value("i", 0)  # 'i' stands for an integer type
lock = Lock()


def process_image(args):
    image_path, target_path, data, attr, idx, image_size = args
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    generate_gt(
        image_name, target_path, data, IMAGE_SIZE=image_size, attr=attr, idx=idx
    )

    # Increment the counter safely using the lock
    with lock:
        counter.value += 1
        if counter.value % 100 == 0:
            print(f"Processed {counter.value} images")


def process_folder(source_folder, target_folder, data, attr, idx, image_size):
    files_to_process = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # If not an image, continue
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                continue
            image_path = os.path.join(root, file)
            image_name = os.path.splitext(file)[0]
            relative_path = os.path.relpath(root, source_folder)
            target_subfolder = os.path.join(target_folder, relative_path)
            # print(target_subfolder)
            if not os.path.exists(target_subfolder):
                os.makedirs(target_subfolder)
            target_path = os.path.join(target_subfolder, image_name + "-GT.png")
            # print(target_path)
            files_to_process.append(
                (image_path, target_path, data, attr, idx, image_size)
            )

    # multiprocessing
    with multiprocessing.Pool(processes=16) as pool:
        pool.map(process_image, files_to_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name",
        type=str,
    )
    parser.add_argument(
        "--source_folder",
        type=str,
    )
    parser.add_argument(
        "--target_folder",
        type=str,
    )
    parser.add_argument(
        "--image_size",
        type=int,
    )
    args = parser.parse_args()
    IMAGE_SIZE = args.image_size
    file_name = args.file_name
    data = read_json(file_name)

    attr_list = [
        "category",
        "color",
        "line_type",
        "line_num",
        "function",
        "bidirection",
        "boundary",
    ]
    cls_id_list = [
        line_cls2id,
        color_cls2id,
        linetype_cls2id,
        linenum_cls2id,
        attribute_cls2id,
        direction_cls2id,
        boundary_cls2id,
    ]

    manager = Manager()
    counter = manager.Value("i", 0)

    processes = []
    for attr, idx in zip(attr_list, cls_id_list):
        print("Process {}".format(attr))
        source_folder = args.source_folder
        target_folder = args.target_folder + attr
        p = multiprocessing.Process(
            target=process_folder,
            args=(source_folder, target_folder, data, attr, idx, args.image_size),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
