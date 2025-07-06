import PIL.Image as Image
import numpy as np
import os
import cv2

# 读取Cities1to30_30cm_BBA_BGRN_22Q1_1303000100033_1_1_0_0_512_512-GT.png
output_dir = 'output-GT-2048'
pic_list=['Cities1to30_30cm_BBA_BGRN_22Q1_1303000011330_1_1_0_1024_512_1536-GT.png',
          'Cities1to30_30cm_BBA_BGRN_22Q1_1303221013230_8_6_1536_1024_2048_1536-GT.png',
          'Cities1to30_30cm_BBA_BGRN_22Q1_1303221013230_8_7_0_512_512_1024-GT.png']
# 创建文件夹。如果文件夹存在，不创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

kernel_size=7
for pic in pic_list:
    img = Image.open(pic)
    img = img.resize((2048, 2048))
    print(img.size)
    # dilate 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.dilate(np.array(img), kernel, iterations=1)
    img = Image.fromarray(img)
    # import pdb; pdb.set_trace()
    # 转为L模式
    # img = img.convert('P')
    # 转为0-255
    img = img.point(lambda i: i * 255)
    # save as png
    img.save(os.path.join(output_dir, 'dilate7'+ pic[:-4]+'2048_L.png'))
