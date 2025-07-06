import os
from PIL import Image
import numpy as np

# 定义文件夹路径
folder_path = '/data/hongbo.zhao/mmsegmentation/data/satellite_instance/ann_dir/val'

# 初始化一个空列表来存储全黑图片的文件名
black_images = []

# 遍历文件夹中的每个文件
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # 加载图像
        img = Image.open(os.path.join(folder_path, filename))
        # 转换为numpy数组
        img_array = np.array(img)
        # 检查是否所有像素都是0（即图像是否全黑）
        if np.all(img_array == 0):
            # 如果图像全黑，删除文件名中的后缀和'-GT'，然后添加到列表中
            black_images.append(filename.replace('.png', '').replace('-GT', ''))

# 保存全黑图像的文件名
with open('black_images_val.txt', 'w') as f:
    for item in black_images:
        f.write("%s\n" % item)

# 打印全黑图像的数量
print(f'Number of black images: {len(black_images)}')

# 打印所有图像的数量
print(f'Number of all images: {len(os.listdir(folder_path))}')