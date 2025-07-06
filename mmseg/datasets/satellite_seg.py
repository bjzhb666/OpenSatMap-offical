from typing import List
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp
import mmengine.fileio as fileio
import mmengine
import json
import torch
import torchvision
from IPython import embed
import torch.distributed as dist
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.special import comb as n_over_k

@ DATASETS.register_module()
class SatelliteSegDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'lines'),
        palette=[[0, 0, 0], [255, 255, 255]]) # 分割调色板
    
    def __init__(self, img_suffix='.png', seg_map_suffix='-GT.png', **kwargs) -> None:

        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        
@ DATASETS.register_module()
class SatelliteInstanceDataset(BaseSegDataset):

    METAINFO = dict(
        # classes=('background', 'solid lane_line', 'dashed lane_line','short dashed lane_line', 'thick solid lane_line','other lane_line',
        #          'parking lot lane_line', 'curb', 'virtual_line'),
        # palette=[[0, 0, 0], [255, 255, 255], [255,0,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [0,255,0], [128,128,128]], # 分割调色板
        # Four class (ignore solid dashed ect.)
        # classes=('background', 'lane_line', 'curb', 'virtual_line'),
        # palette=[[0, 0, 0], [255, 255, 255], [255,0,0], [0,0,255]], # 分割调色板
        # Potential classes for zoom 19, ignore 'thick solid lane_line','other lane_line',
        # classes=('background', 'solid lane_line', 'dashed lane_line','short dashed lane_line',
        #          'parking lot lane_line', 'curb', 'virtual_line'),
        # palette=[[0, 0, 0], [255, 255, 255], [255,0,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255]], # 分割调色板
        # move parking to others
        classes=('background', 'solid lane_line', 'dashed lane_line','short dashed lane_line', 'thick solid lane_line','other lane_line',
                 'curb', 'virtual_line'),
        palette=[[0, 0, 0], [255, 255, 255], [255,0,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [0,255,0]], # 分割调色板
        color_classes=('background', 'white', 'yellow', 'others', 'none'),
        color_palette=[[0, 0, 0], [255, 255, 255], [255, 255, 0], [0, 0, 255], [0, 255, 255]],
        line_type_classes=('background','导流区', '实线', '虚线', '停车位', '短粗虚线', '粗实线', '其他', '待转区', '引导线', '无'),
        line_type_palette=[[0, 0, 0], [255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                            [255, 0, 255], [0, 255, 255], [0, 0, 128], [255, 255, 255], [0, 128, 0]],
        line_num_classes=('background','单线', '双线', '其他', '无'),
        line_num_palette=[[0, 0, 0], [255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
        attribute_classes=('background','无', '禁停网格', '减速车道', '公交车道', '其他', '潮汐车道', '借道区', '可变车道'),
        attribute_palette=[[0, 0, 0], [255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], 
                 [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 128, 128]],
        whether_bidirection_classes=('background', '无', '双向'),
        whether_bidirection_palette=[[0, 0, 0],[255, 165, 0], [255, 0, 0]],
        whether_boundary_classes=('background', 'yes', 'no'),
        whether_boundary_palette=[[0, 0, 0],[255, 165, 0], [255, 0, 0]]
    )

    
    def __init__(self, direction_path, color_path, line_type_path, line_num_path, 
                 attribute_path, ifbidirection_path, ifboundary_path,
                 img_suffix='.png', seg_map_suffix='-GT.png',
                 direction_map_suffix='-GT.png',  **kwargs) -> None:

        self.direction_path = direction_path
        self.color_path = color_path
        self.line_type_path = line_type_path
        self.line_num_path = line_num_path
        self.attribute_path = attribute_path
        self.ifbidirection_path = ifbidirection_path
        self.ifboundary_path = ifboundary_path

        self.direction_map_suffix = direction_map_suffix
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        direction_dir = osp.join(self.data_root, self.direction_path)
        color_dir = osp.join(self.data_root, self.color_path)
        line_type_dir = osp.join(self.data_root, self.line_type_path)
        line_num_dir = osp.join(self.data_root, self.line_num_path)
        attribute_dir = osp.join(self.data_root, self.attribute_path)
        ifbidirection_dir = osp.join(self.data_root, self.ifbidirection_path)
        ifboundary_dir = osp.join(self.data_root, self.ifboundary_path)

        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else: # NOTE: not given the ann_file (ours)
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                if direction_dir is not None:
                    direction_map = img[:-_suffix_len] + self.direction_map_suffix
                    data_info['direction_path'] = osp.join(direction_dir, direction_map)
                if color_dir is not None:
                    color_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['color_path'] = osp.join(color_dir, color_map)
                if line_type_dir is not None:
                    line_type_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['line_type_path'] = osp.join(line_type_dir, line_type_map)
                if line_num_dir is not None:
                    line_num_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['line_num_path'] = osp.join(line_num_dir, line_num_map)
                if attribute_dir is not None:
                    attribute_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['attribute_path'] = osp.join(attribute_dir, attribute_map)
                if ifbidirection_dir is not None:
                    ifbidirection_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['ifbidirection_path'] = osp.join(ifbidirection_dir, ifbidirection_map)
                if ifboundary_dir is not None:
                    ifboundary_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['ifboundary_path'] = osp.join(ifboundary_dir, ifboundary_map)

                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

