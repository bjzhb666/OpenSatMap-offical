import os
import shutil
import argparse

def load_file_list(file_path):
    """
    Load image name list from a text file
    """
    with open(file_path, "r") as f:
        return set(line.strip() for line in f.readlines())

def move_images(root_dir, train_file, val_file):
    """
    Move images to corresponding train and val folders based on train.txt and val.txt
    """
    # Get all subdirectories in the root directory
    base_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Read train.txt and val.txt
    train_images = load_file_list(train_file)
    val_images = load_file_list(val_file)

    # Iterate through each directory and move files
    for base_dir in base_dirs:
        base_dir_path = os.path.join(root_dir, base_dir)
        use_dir = os.path.join(base_dir_path, "use")
        train_dir = os.path.join(base_dir_path + 'trainval', "train")
        val_dir = os.path.join(base_dir_path+ 'trainval', "val")
        
        # Create train and val folders
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        if not os.path.exists(use_dir):
            print(f"Skipping {use_dir}, directory does not exist.")
            continue
        
        # Move files
        for image_name in os.listdir(use_dir):
            # import pdb; pdb.set_trace()
            image_path = os.path.join(use_dir, image_name)
            if image_name.replace('-GT','')  in train_images or image_name.replace('-GT-direction','') in train_images:
                pass
                # shutil.move(image_path, os.path.join(train_dir, image_name))
            elif image_name.replace('-GT','')  in val_images or image_name.replace('-GT-direction','') in val_images:
                pass
                # shutil.move(image_path, os.path.join(val_dir, image_name))

    print("File moving completed!")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Move image files based on train.txt and val.txt.")
    parser.add_argument("root_dir", type=str, help="Root directory path")
    parser.add_argument("train_file", type=str, help="Path to train.txt file")
    parser.add_argument("val_file", type=str, help="Path to val.txt file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute file moving
    move_images(args.root_dir, args.train_file, args.val_file)

if __name__ == "__main__":
    main()
