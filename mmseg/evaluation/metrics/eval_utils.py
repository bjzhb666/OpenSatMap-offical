from mmengine.utils import mkdir_or_exist
from PIL import Image
import os.path as osp
import numpy as np
import cv2
import torch
import pycocotools.mask as maskUtils
from collections import Counter

def save_prediction(data_sample, pred_label, pred_line_type_label, pred_direct_map_ori, save_dir):
    img_name = osp.basename(data_sample['img_path'])[:-4]
    pred_label_cpu = pred_label.cpu().numpy()
    
    mkdir_or_exist(f'{save_dir}')
    # save the prediction
    # save pred_label as png
    output_png = Image.fromarray(pred_label_cpu.astype(np.uint8))
    output_png.save(f'{save_dir}/pred_label-{img_name}.png')
    
    if pred_line_type_label is not None:
        pred_line_type_label_cpu = pred_line_type_label.cpu().numpy()
        output_line_type_png = Image.fromarray(pred_line_type_label_cpu.astype(np.uint8))
        output_line_type_png.save(f'{save_dir}/pred_line_type-{img_name}.png')
        gt_line_type_map = data_sample['gt_line_type_map']['data'].squeeze().cpu().numpy()
        gt_line_type_png = Image.fromarray(gt_line_type_map.astype(np.uint8))
        gt_line_type_png.save(f'{save_dir}/gt_line_type-{img_name}.png')
        
    if pred_direct_map_ori is not None:
    # pred_line_num_label_cpu = pred_line_num_label.cpu().numpy()
        np.save(f'{save_dir}/pred_direct_map_512-{img_name}.npy', pred_direct_map_ori.cpu().numpy())
        gt_direction = data_sample['direction_map']['data'].squeeze().cpu().numpy()
        np.save(f'{save_dir}/gt_direction-{img_name}.npy', gt_direction)
    # pred_direct_map_512_cpu = pred_direct_map_512.cpu().numpy()
    # pred_tag_map_512_cpu = pred_tag_map_512.cpu().numpy()
    
    gt_seg = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
    gt_instance_map = data_sample['gt_instance_map']['data'].squeeze().cpu().numpy()
    # gt_line_num_map = data_sample['gt_line_num_map']['data'].squeeze().cpu().numpy()
    # output_line_num_png = Image.fromarray(pred_line_num_label_cpu.astype(np.uint8))
    gt_seg_png = Image.fromarray(gt_seg.astype(np.uint8))
    # gt_line_num_png = Image.fromarray(gt_line_num_map.astype(np.uint8))
    gt_instance_png = Image.fromarray(gt_instance_map.astype(np.uint8))
    # output_line_num_png.save(f'{save_dir}/pred_line_num-{img_name}.png')
    gt_seg_png.save(f'{save_dir}/gt_seg-{img_name}.png')
    # gt_line_num_png.save(f'{save_dir}/gt_line_num-{img_name}.png')
    gt_instance_png.save(f'{save_dir}/gt_instance-{img_name}.png')
    # np.save(f'{save_dir}/pred_tag_map_512-{img_name}.npy', pred_tag_map_512_cpu)
    


def sample_from_positions(position, with_noise=True):
    x, y = position[:, 0], position[:, 1]
    if with_noise:
        percentiles = np.linspace(1, 99, 50)
    else:
        percentiles = np.linspace(0, 100, 50)
    x_quantiles = np.percentile(x, percentiles, interpolation='nearest')
    y_quantiles = np.percentile(y, percentiles, interpolation='nearest')

    sampled_points = []  # [x, y]

    if np.mean(np.abs(np.diff(x_quantiles))) >= np.mean(np.abs(np.diff(y_quantiles))):
        for quantile_value in x_quantiles:
            index = np.where(np.isclose(x, round(quantile_value, 0)))[0]
            assert len(index) > 0
            sampled_points.append(
                [int(round(quantile_value, 0)), int(round(np.percentile(y[index], 50), 0))])
    else:
        for quantile_value in y_quantiles:
            index = np.where(np.isclose(y, round(quantile_value, 0)))[0]
            assert len(index) > 0
            sampled_points.append(
                [int(round(np.percentile(x[index], 50), 0)), int(round(quantile_value, 0))])

    return np.array(sampled_points)

def generate_line_mask_without_direction(coordinates, line_width, image_size):
    """
    生成线的掩码（二值掩码），不包含方向
    :param coordinates: 线的坐标点
    :param line_width: 线的宽度
    :param image_size: 图像大小
    :return: 线的掩码
    """
    # 如果坐标点少于2个，返回None
    if len(coordinates) < 2:
        return None

    # 创建一个空白图像作为掩码
    mask = np.zeros(image_size, dtype=np.uint8)

    # 将坐标点转换为整数
    coordinates = np.array(coordinates, dtype=np.int32)

    # 绘制连续线段
    cv2.polylines(mask, [coordinates], False, 255, thickness=line_width)

    # 找到线的内部点并返回
    # indices = np.where(mask == value)
    # line_points = np.column_stack((indices[1], indices[0]))

    return mask

def get_instance_image(position, h, w, y_pred):
    new_image = np.zeros([h, w], dtype=np.uint8)
    new_image[position[:, 1], position[:, 0]] = y_pred
    return new_image


def get_position(binary_mask):
    h, w = binary_mask.shape
    x = torch.arange(w).unsqueeze(0).repeat(h, 1)[binary_mask]
    y = torch.arange(h).unsqueeze(1).repeat(1, w)[binary_mask]
    return torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1).numpy()


def show_instance_map(mask):
    def generate_colormap(num_colors=1000):
        # 初始化colormap列表
        colormap = [[0, 0, 0]]

        # HSV颜色空间转换为RGB
        def hsv_to_rgb(h, s, v):
            c = v * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c
            if h < 60:
                return (c + m, x + m, m)
            elif h < 120:
                return (x + m, c + m, m)
            elif h < 180:
                return (m, c + m, x + m)
            elif h < 240:
                return (m, x + m, c + m)
            elif h < 300:
                return (x + m, m, c + m)
            else:
                return (c + m, m, x + m)

        # 生成colormap
        for i in range(1, num_colors):
            hue = (i / num_colors) * 360  # 色相在0到360度之间变化
            saturation = 0.8  # 饱和度保持较高
            value = i / num_colors * 80 + 55  # 明度保持在60%到80%之间变化

            rgb = hsv_to_rgb(hue, saturation, value)

            # 四舍五入并转换为整数，以便直接用于像素值
            colormap.append([int(round(c * 255)) for c in rgb])

        return np.array(colormap)


    def colorful(mask, colormap):
        color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
        for i in np.unique(mask):
            color_mask[mask == i] = colormap[i]

        return np.uint8(color_mask)

    colormap = generate_colormap()
    return colorful(mask, colormap)


def debug_instance_pred(data_sample, item, instance_output_dir, dt_instance_ori):
    ori_img = Image.open(data_sample['img_path']).convert('RGB')
    img_name = osp.basename(data_sample['img_path'])[:-4]

    def black_to_white(img):
        black_pixels = np.all(img == [0, 0, 0], axis=-1)
        img[black_pixels] = np.array([235, 235, 235])
        return img

    gt_instance = Image.fromarray(black_to_white(show_instance_map(item["gt_instance"])))
    dt_instance = Image.fromarray(black_to_white(show_instance_map(item["dt_instance"])))
    dt_instance_ori = Image.fromarray(black_to_white(show_instance_map(dt_instance_ori)))

    save_dir = instance_output_dir
    mkdir_or_exist(save_dir)

    ori_img.save(osp.join(save_dir, f"{img_name}_img.jpg"))
    # gt_seg.save(osp.join(save_dir, f"{img_name}_gtseg.jpg"))
    # dt_seg.save(osp.join(save_dir, f"{img_name}_dtseg.jpg"))
    gt_instance.save(osp.join(save_dir, f"{img_name}_gtins.jpg"))
    dt_instance.save(osp.join(osp.join(save_dir, f"{img_name}_dtins.jpg")))
    dt_instance_ori.save(osp.join(osp.join(save_dir, f"{img_name}_dtinsori.jpg")))


def merge_dicts_in_tuple(tuple_of_dicts):
    cnt = 1
    merged_dict = {
        'coco_gt': {
            'annotations': [],
            'images': [],
        },
        'coco_dt': [],
    }
    
    for d in tuple_of_dicts:
        for i in range(len(d['coco_gt']['annotations'])):
            d['coco_gt']['annotations'][i]['id'] = cnt
            cnt += 1

        merged_dict['coco_gt']['annotations'].extend(d['coco_gt']['annotations'])
        merged_dict['coco_gt']['images'].extend(d['coco_gt']['images'])
        merged_dict['coco_dt'].extend(d['coco_dt'])
    
    return merged_dict


def prepare_coco_dict(data_sample, item, use_seg_GT=False):
        # coco_dict is used for instance evaluation
    coco_dict = {
        'coco_gt': {
            'annotations': [],
            'images': [],
        },
        'coco_dt': [],
    }

    coco_dict['coco_gt']['images'].append({
        'id': data_sample['img_path'].split('/')[-1],
        'width': data_sample['ori_shape'][1],
        'height': data_sample['ori_shape'][0],
    })

    for instance_id in np.unique(item["gt_instance"]):
        if instance_id == 0: continue
        category_id, _ = Counter(item["gt_label"][item["gt_instance"] == instance_id]).most_common(1)[0]
        mask = (item["gt_instance"] == instance_id).astype(np.uint8)
        if category_id == 0: 
            print(data_sample['img_path'])
            print(Counter(item["gt_label"][item["gt_instance"] == instance_id]))
            # continue
        assert category_id != 0

        rle = maskUtils.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        for k in rle.keys():
            if isinstance(rle[k], bytes):
                rle[k] = rle[k].decode()

        sampled_points = sample_from_positions(get_position(mask.astype(bool)), with_noise=False)
        assert len(sampled_points) > 0
        
        coco_dict['coco_gt']['annotations'].append({
            # 'id': len(coco_dict['coco_gt']['annotations']) + 1,
            'image_id': data_sample['img_path'].split('/')[-1],
            'category_id': int(category_id),
            'segmentation': rle,
            'bbox': maskUtils.toBbox(rle).tolist(),
            'iscrowd': 0,
            'sampled_points': sampled_points.tolist(),
            'area': np.count_nonzero(mask.astype(bool)),
        })

    for instance_id in np.unique(item["dt_instance"]):
        scores = item["seg_probs"][1:, item["dt_instance"] == instance_id]
        if use_seg_GT:
            scores = np.ones_like(scores)
            # scores = np.random.uniform(0.95, 1, scores.shape)
        score = np.mean(np.max(scores, axis=0))
        cnt = Counter(item["dt_label"][item["dt_instance"] == instance_id]).most_common(2)
        category_id = cnt[0][0]
        if category_id == 0 and instance_id != 0:
            category_id = cnt[1][0]

        mask = (item["dt_instance"] == instance_id).astype(np.uint8)
        rle = maskUtils.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        for k in rle.keys():
            if isinstance(rle[k], bytes):
                rle[k] = rle[k].decode()

        sampled_points = sample_from_positions(get_position(mask.astype(bool)), with_noise=False)
        assert len(sampled_points) > 0
        coco_dict['coco_dt'].append({
            "image_id": data_sample['img_path'].split('/')[-1],
            "category_id": int(category_id),
            "segmentation": rle,
            'bbox': maskUtils.toBbox(rle).tolist(),
            "score": float(score),
            "sampled_points": sampled_points.tolist(),
        })

    return coco_dict
