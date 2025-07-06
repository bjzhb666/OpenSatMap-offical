import cv2
import numpy as np
from .eval_utils import (get_position, get_instance_image, sample_from_positions, 
                         generate_line_mask_without_direction, debug_instance_pred, prepare_coco_dict)


def watershed(binary_image, position):
    binary_image = (binary_image * 255).astype(np.uint8)
    # binary_image = cv2.GaussianBlur(binary_image, (3, 3), 0)
    kernel = np.ones((1, 1), np.uint8)
    # import pdb; pdb.set_trace()
    if binary_image.shape[0] == 512:
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=3)
        # sure_bg = cv2.dilate(opening, kernel, iterations=5)
        sure_bg = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    elif binary_image.shape[0] == 1024:
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=3)
        # sure background area
        sure_bg = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers)
    y_pred = np.array([markers[pos[1], pos[0]] for pos in position])

    return y_pred - 1

def watercluster(label, pred_label, seg_probs, gt_instance, data_sample, 
                 GT_without_Water, save_instance_pred, use_seg_GT, minimal_area=1, 
                 num_classes=4, dilate_kernel=0, instance_output_dir=None):
    per_class_gt_label = [(label == index).bool() for index in list(range(1, num_classes))] # list of [H, W], len=3
    per_class_dt_label = [(pred_label == index).bool() for index in list(range(1, num_classes))] # list of [H, W]

    assert dilate_kernel != 0

    gt_positions = [get_position(gt_mask.cpu().numpy()) for gt_mask in per_class_gt_label] # list of [N, 2], len=3, N is the number of points in the mask class i
    dt_positions = [get_position(dt_mask.cpu().numpy()) for dt_mask in per_class_dt_label] # [N, 2]
    dt_instance = np.zeros_like(gt_instance.cpu().numpy()) # [H, W]
    dt_instance_ori = np.zeros_like(gt_instance.cpu().numpy()) # [H, W], only for drawing
    
    for index, (dt_mask, dt_position, gt_mask, gt_position) in enumerate(zip(per_class_dt_label, dt_positions, per_class_gt_label, gt_positions)):
        # dt_position, gt_position are the positions of the points in class i
        # get instance predictions via watershed
        y_pred = watershed(dt_mask.cpu().numpy(), dt_position)
        y_pred[y_pred < 0] = 0

        # for dt_instance_ori, only for drawing
        instance_id_start = np.max(dt_instance)
        ori = get_instance_image(dt_position, dt_instance_ori.shape[0], dt_instance_ori.shape[1], y_pred)
        ori = ori + (ori != 0) * instance_id_start
        dt_instance_ori = dt_instance_ori + ori
        
        if GT_without_Water:
            y_pred = np.array(gt_instance[gt_position[:, 1], gt_position[:, 0]].cpu().numpy(), dtype=np.int32)
            dt_position = gt_position
            dt_mask = gt_mask
        # sample and reconstruct
        for instance_id in np.unique(y_pred):
            if instance_id == 0: continue

            instance_position = dt_position[y_pred == instance_id]
            # remove small regions
            if len(instance_position) <= minimal_area: continue # 面积小于100的实例不要

            sampled_positions = sample_from_positions(instance_position)
            rec = generate_line_mask_without_direction(
                coordinates=sampled_positions,
                line_width=1,
                image_size=dt_mask.cpu().numpy().shape,
            )
            rec = cv2.dilate(rec, kernel=np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8), iterations=1)
            # removing overlap
            if np.sum((rec == 255) * dt_instance) > 0: continue
            dt_instance = dt_instance + (rec == 255) * (instance_id + instance_id_start)

    item = {
        "seg_probs": seg_probs.cpu().numpy(),
        "gt_label": label.cpu().numpy(),
        "dt_label": pred_label.cpu().numpy(),
        "gt_instance": gt_instance.cpu().numpy(),
        "dt_instance": dt_instance,
        # "dt_instance": gt_instance.cpu().numpy(),
    }

    if save_instance_pred:
        debug_instance_pred(data_sample, item, instance_output_dir, dt_instance_ori=dt_instance_ori)

    coco_dict = prepare_coco_dict(data_sample, item, use_seg_GT)

    return coco_dict

