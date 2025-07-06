# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable


from pycocotools.coco import COCO
from .coco_eval import COCOeval

from mmseg.registry import METRICS
from .eval_utils import (save_prediction, merge_dicts_in_tuple)
from .cluster import watercluster


@METRICS.register_module()
class InstanceIoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        save_ori_prediction (bool): Save the original prediction mask.
            Defaults to False.
        use_seg_GT (bool): Use the segmentation ground truth as the input of the
            watershed algorithm. Defaults to False. (only for debugging)
        GT_without_Water (bool): Use the instance ground truth as the input of the sampling algorithm.
            Defaults to False. (only for debugging)
        save_instance_pred (bool): Save the instance prediction mask. (instance before sampling, 
            instance after sampling, GT instance and original image) Defaults to False. (only for debugging)
        minimal_area (int): The minimal area of the instance mask. Defaults to 100.
        chamfer_thrs (list[int]): The threshold of chamfer distance. Defaults to [3, 10, 15].
        dilate_kernel_size (int): The kernel size of the dilate operation. Defaults to 3.   
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 save_ori_prediction: bool = False,
                 use_seg_GT: bool = False,
                 GT_without_Water: bool = False,
                 save_instance_pred: bool = False,
                 minimal_area: int = 100,
                 chamfer_thrs: List[int] = [3, 5, 10, 15],
                 dilate_kernel_size: int = 3,
                 instance_dir: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.save_ori_prediction = save_ori_prediction
        self.use_seg_GT = use_seg_GT    
        self.GT_without_Water = GT_without_Water
        self.save_instance_pred = save_instance_pred
        self.minimal_area = minimal_area
        self.chamfer_thrs = chamfer_thrs
        self.dilate_kernel_size = dilate_kernel_size
        self.instance_dir = instance_dir
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
       
        
        num_classes = len(self.dataset_meta['classes'])
        num_line_type_classes = len(self.dataset_meta['line_type_classes'])
        has_line_type = 'pred_seg_line_type' in data_samples[0]
        has_direct_map = 'pred_direct_map_2048' in data_samples[0]

        for image_id, data_sample in enumerate(data_samples):
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            
            if has_direct_map:
                pred_direct_map_2048 = data_sample['pred_direct_map_2048']['data']
                # # add one extra dimension to use F.interpolate
                pred_direct_map_2048 = pred_direct_map_2048.unsqueeze(1)

                # # resize to original size
                pred_direct_map_ori = F.interpolate(pred_direct_map_2048, size=data_sample['ori_shape'], mode='bilinear', align_corners=False)
                
                # # remove the extra dimension
                pred_direct_map_ori = pred_direct_map_ori.squeeze(1).squeeze() # ori_H, ori_W
            else:
                pred_direct_map_ori = None
                
            if 'pred_seg_line_type' in data_sample:
                pred_line_type_label = data_sample['pred_seg_line_type']['data'].squeeze()
            else:
                pred_line_type_label = None

            if self.save_ori_prediction:
                save_prediction(data_sample, pred_label, pred_line_type_label, pred_direct_map_ori, save_dir=osp.join(self.output_dir, 'instance'))
            # format_only always for test dataset without ground truth
            if not self.format_only:
                seg_logits = data_sample['seg_logits']['data']  # [num_classes, H, W]
                seg_probs = F.softmax(seg_logits, dim=0) # [num_classes, H, W]
                label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label) # [H, W]
                
                gt_instance = data_sample['gt_instance_map']['data'].squeeze()  # [H, W]

                # line_type_logits = data_sample['pred_seg_line_type']['data']  # [num_line_type_classes, H, W]
                # line_type_probs = F.softmax(line_type_logits, dim=0) # [num_line_type_classes, H, W]
                line_type_label = data_sample['gt_line_type_map']['data'].squeeze().to(pred_line_type_label) # (H, W)
                
                if self.use_seg_GT:
                    pred_label = label
                    pred_line_type_label = line_type_label

                coco_dict = watercluster(label, pred_label, seg_probs, gt_instance, data_sample, self.GT_without_Water,
                                         self.save_instance_pred, self.use_seg_GT, self.minimal_area, num_classes=num_classes, 
                                         dilate_kernel=self.dilate_kernel_size, instance_output_dir=self.instance_dir)

                combine_tuple = self.intersect_and_union(pred_label, label, num_classes, self.ignore_index) 
                if pred_line_type_label is not None:
                    combine_tuple += self.intersect_and_union(pred_line_type_label, line_type_label, num_line_type_classes, self.ignore_index)
                combine_tuple += (coco_dict, )
                
                self.results.append(combine_tuple)
                
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                gt_seg = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                    gt_seg = gt_seg + 1
                    if has_line_type:
                        output_line_type_mask = output_line_type_mask + 1
        
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)
                output_gt = Image.fromarray(gt_seg.astype(np.uint8))
                output_gt.save(osp.abspath(osp.join(self.output_dir, f'{basename}_gt.png')))
                
                if has_line_type:
                    png_line_type_filename = osp.abspath(
                        osp.join(self.output_dir, f'{basename}_line_type.png'))
                    output_line_type_mask = pred_line_type_label.cpu().numpy()
                    output_line_type = Image.fromarray(output_line_type_mask.astype(np.uint8))
                    output_line_type.save(png_line_type_filename)
                

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])

        results = tuple(zip(*results))
        results_len = len(results)
        assert len(results) in [9, 5]
        
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        if results_len == 9:
            total_area_intersect_line_type = sum(results[4])
            total_area_union_line_type = sum(results[5])
            total_area_pred_line_type_label = sum(results[6])
            total_area_line_type_label = sum(results[7])


            coco_dict = results[8] # len(results[12])==number of images
        else: # len=5
            coco_dict = results[4] 

        total_coco_dict = merge_dicts_in_tuple(coco_dict)

        # debug code
        # img_ids = [img['id'] for img in total_coco_dict['coco_gt']['images']]
        # img_ids = np.unique(np.array(img_ids))
        # print(len(img_ids))
        #
        # img_ids_gt = [ann['image_id'] for ann in total_coco_dict['coco_gt']['annotations']]
        # img_ids_gt = np.unique(np.array(img_ids_gt))
        # print(len(img_ids_gt))
        #
        # img_ids_dt = [dt['image_id'] for dt in total_coco_dict['coco_dt']]
        # img_ids_dt = np.unique(np.array(img_ids_dt))
        # print(len(img_ids_dt))
        
        # Instance Evaluation via COCO API
        # print('********************')
        # print(np.max(np.array([coco['area'] for coco in self.coco['coco_gt']['annotations']])))
        class_names = self.dataset_meta['classes']
        _coco_api = COCO()
        _coco_api.dataset = {
            "categories": [{"id": i, "name": class_name} for i, class_name in enumerate(class_names) if i != 0],
            "images": total_coco_dict['coco_gt']['images'],
            "annotations": total_coco_dict['coco_gt']['annotations'],
        }
        _coco_api.createIndex()
        coco_dt = _coco_api.loadRes(total_coco_dict['coco_dt']) 
        coco_eval = COCOeval(_coco_api, coco_dt, iouType='segm', threshold=[0.5, 0.75])  # 'segm' 表示实例分割评估

        iou_begin, iou_end = .5, .95
        coco_eval.params.iouThrs = np.linspace(
            iou_begin, iou_end, int(np.round((iou_end - iou_begin) / .05)) + 1, endpoint=True,
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval_chamfer = COCOeval(_coco_api, coco_dt, iouType='line', threshold=self.chamfer_thrs)  # 'line' 表示线分割评估
        coco_eval_chamfer.params.iouThrs = np.array(self.chamfer_thrs) # 单位是像素
        coco_eval_chamfer.evaluate()
        coco_eval_chamfer.accumulate()
        coco_eval_chamfer.summarize()
        

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        if results_len == 9:
            ret_metrics_line_type = self.total_area_to_metrics(
                total_area_intersect_line_type, total_area_union_line_type, total_area_pred_line_type_label,
                total_area_line_type_label, self.metrics, self.nan_to_num, self.beta, metric_suffix='_line_type')
            
            line_type_class_name = self.dataset_meta['line_type_classes']

            ret_metrics_summary_line_type = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics_line_type.items()
            })

            metrics_line_type = dict()
            # import pdb; pdb.set_trace()
            for key, val in ret_metrics_summary_line_type.items():
                if key == 'aAcc_line_type':
                    metrics_line_type[key] = val
                else:
                    metrics_line_type['m' + key] = val
            
            # each class table
            ret_metrics_line_type.pop('aAcc_line_type', None)
            ret_metrics_class_line_type = OrderedDict({
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics_line_type.items()
            })
            ret_metrics_class_line_type.update({'Class': line_type_class_name})
            ret_metrics_class_line_type.move_to_end('Class', last=False)
            class_table_data_line_type = PrettyTable()
            for key, val in ret_metrics_class_line_type.items():
                class_table_data_line_type.add_column(key, val)

            print_log('per class results:', logger)
            print_log('\n' + class_table_data_line_type.get_string(), logger=logger)
            metrics.update(metrics_line_type)
        
        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1,
                              metric_suffix:str = ''):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
            metric_suffix (str): The suffix of the metric name. Default: ''.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric+metric_suffix: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
