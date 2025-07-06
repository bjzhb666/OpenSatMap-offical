import numpy as np
import os
import glob
from IPython import embed
import cv2

# 读取当前路径下以gt_direction开头的文件名
file_folder = 'work_dirs/segnext_instance_debugtrain/'

all_files = os.listdir(file_folder)

# gt_direction文件名
direction_files = [f for f in all_files if f.startswith('gt_direction')]

# pred_direction文件名
pred_direction_files = [f.replace('gt_direction','pred_direct_map_512') for f in direction_files]

# gt_seg文件名
seg_files = [f.replace('gt_direction','gt_seg').replace('npy' ,'png') for f in direction_files]

# pred_seg文件名
pred_seg_files = [f.replace('gt_direction','pred_label').replace('npy' ,'png') for f in direction_files]

# ae_gt文件名
ae_gt_files = [f.replace('gt_direction','gt_instance').replace('npy' ,'png') for f in direction_files]

# ae_pred文件名
ae_pred_files = [f.replace('gt_direction','pred_tag_map_512') for f in direction_files]

diff_total = 0
diff_count = 0

for i in range(len(direction_files)):
    # print(f'{i}: {direction_files[i]}')
    # # 读取gt_direction文件
    gt_dir = np.load( file_folder+direction_files[i])
    # 读取gt_pred_map_512文件
    gt_pred = np.load(file_folder+pred_direction_files[i])

    # 读取gt_seg文件
    gt_seg = cv2.imread(file_folder+seg_files[i], cv2.IMREAD_GRAYSCALE)
    # 读取pred_seg文件
    pred_seg = cv2.imread(file_folder+pred_seg_files[i], cv2.IMREAD_GRAYSCALE)

    # 读取ae_gt文件
    ae_gt = cv2.imread(file_folder+ae_gt_files[i], cv2.IMREAD_GRAYSCALE)
    # 读取ae_pred文件
    ae_pred = np.load(file_folder+ae_pred_files[i], cv2.IMREAD_GRAYSCALE)

    # 前景位置
    front = (gt_seg != 0) & (gt_seg != 100)

    # direction做差
    real_gt_dir = gt_dir[front]
    real_pred_dir = gt_pred.squeeze()[front]
    diff = real_gt_dir - real_pred_dir

    if diff.size == 0: # remove all background pic
        continue
    # 把diff取绝对值
    diff = np.abs(diff)
    # print('# angle diff < 0.1:', np.sum(diff<0.1)/diff.size)
    # print('# angle diff < 0.314:', np.sum(diff<0.314)/diff.size)
    # print('# angle diff < 0.628:', np.sum(diff<0.628)/diff.size)
    # print('# angle diff < 0.942:', np.sum(diff<0.942)/diff.size)
    # 取平均值
    diff_total += np.sum(diff.size)
    diff_count += np.sum(diff<0.1)

print('average diff:', diff_count/diff_total)

