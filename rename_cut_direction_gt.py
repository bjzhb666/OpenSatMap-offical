import os
import re

# 指定文件夹路径
folder_path = 'data/satellite_instance/direction/val'

# 匹配如 '...-GT-direction...' 结构的正则表达式模式
pattern = re.compile(r'(.+)-GT-direction(.+)\.png')

# 遍历文件夹内所有文件
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        # 重构文件名
        new_filename = f"{match.group(1)}{match.group(2)}-GT-direction.png"
        # 获取旧文件和新文件的完整路径
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        # 重命名文件
        os.rename(old_file_path, new_file_path)

print("所有文件重命名完成！")