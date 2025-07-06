import os
import shutil
import argparse
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert satellite dataset to mmsegmentation format')
    parser.add_argument('--txt_file_path', help='satellite folder path', )
    parser.add_argument('--source_folder_path', help='output path',)
    parser.add_argument('--target_folder_path', help='output path',)
    
    args = parser.parse_args()
    return args

def move_file(line, source_folder_path, target_folder_path):
    filename = line.strip()
    source_file_path_GT = os.path.join(source_folder_path, filename + '-GT.png')
    source_file_path = os.path.join(source_folder_path, filename + '.png')
    source_file_path_direction = os.path.join(source_folder_path, filename + '-GT-direction.png')

    if os.path.isfile(source_file_path):
        shutil.move(source_file_path, target_folder_path)
        return 1
    elif os.path.isfile(source_file_path_GT):
        shutil.move(source_file_path_GT, target_folder_path)
        return 1
    elif os.path.isfile(source_file_path_direction):
        shutil.move(source_file_path_direction, target_folder_path)
        return 1
    return 0

def main():
    args = parse_args()
    text_file_path = args.txt_file_path
    source_folder_path = args.source_folder_path
    target_folder_path = args.target_folder_path

    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    with open(text_file_path, 'r') as file:
        lines = file.readlines()

    with multiprocessing.Pool() as pool:
        results = pool.starmap(move_file, [(line, source_folder_path, target_folder_path) for line in lines])

    moved_count = sum(results)
    print(f"Moved a total of {moved_count} files to the {target_folder_path} folder.")

if __name__ == '__main__':
    main()
