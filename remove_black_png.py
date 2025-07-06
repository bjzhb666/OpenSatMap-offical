import os
import shutil

# 请将下面的路径替换为实际的文件路径
text_file_path = 'black_images_train.txt'      # 文本文件路径
source_folder_path = 'data/satellite_instance/direction/train'  # 源文件夹路径
target_folder_path = 'data/satellite_instance/direction/not_use_train'  # 目标文件夹路径

moved_count = 0  # 移动文件的计数器

# 创建target_folder_path文件夹
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

# 打开文本文件并逐行读取
with open(text_file_path, 'r') as file:
    for line in file:
        # import pdb; pdb.set_trace()
        # 去除文件名的前后空格和换行符
        filename = line.strip()
        # 构建源文件路径
        # source_file_path = os.path.join(source_folder_path, filename + '-GT.png')
        # source_file_path = os.path.join(source_folder_path, filename + '.png')
        source_file_path = os.path.join(source_folder_path, filename + '-GT-direction.png')

        # 检查文件是否存在
        if os.path.isfile(source_file_path):
            # 移动文件到目标文件夹
            shutil.move(source_file_path, target_folder_path)
            moved_count += 1

# 打印移动文件的数量以确认操作
print(f"共移动了 {moved_count} 个文件到 {target_folder_path} 文件夹。")