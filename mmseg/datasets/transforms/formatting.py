# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample
import pdb


@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackInstanceSegInputs(BaseTransform):
    """Pack the inputs data for the instance segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()

        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_color_map' in results:
            if len(results['gt_color_map'].shape) == 2:
                data = to_tensor(results['gt_color_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'color map, usually the color '
                              'map is 2D, but got '
                              f'{results["gt_color_map"].shape}')
                data = to_tensor(results['gt_color_map'].astype(np.int64))
            gt_color_data = dict(data=data)
            data_sample.set_data(dict(gt_color_map=PixelData(**gt_color_data)))

        if 'gt_line_type_map' in results:
            if len(results['gt_line_type_map'].shape) == 2:
                data = to_tensor(results['gt_line_type_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'line type map, usually the line type '
                              'map is 2D, but got '
                              f'{results["gt_line_type_map"].shape}')
                data = to_tensor(results['gt_line_type_map'].astype(np.int64))
            gt_line_type_data = dict(data=data)
            data_sample.set_data(dict(gt_line_type_map=PixelData(**gt_line_type_data)))

        if 'gt_line_num_map' in results:
            if len(results['gt_line_num_map'].shape) == 2:
                data = to_tensor(results['gt_line_num_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'line num map, usually the line num '
                              'map is 2D, but got '
                              f'{results["gt_line_num_map"].shape}')
                data = to_tensor(results['gt_line_num_map'].astype(np.int64))
            gt_line_num_data = dict(data=data)
            data_sample.set_data(dict(gt_line_num_map=PixelData(**gt_line_num_data)))

        if 'gt_attribute_map' in results:
            if len(results['gt_attribute_map'].shape) == 2:
                data = to_tensor(results['gt_attribute_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'attribute map, usually the attribute '
                              'map is 2D, but got '
                              f'{results["gt_attribute_map"].shape}')
                data = to_tensor(results['gt_attribute_map'].astype(np.int64))
            gt_attribute_data = dict(data=data)
            data_sample.set_data(dict(gt_attribute_map=PixelData(**gt_attribute_data)))

        if 'gt_ifbidirection_map' in results:
            if len(results['gt_ifbidirection_map'].shape) == 2:
                data = to_tensor(results['gt_ifbidirection_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'ifbidirection map, usually the ifbidirection '
                              'map is 2D, but got '
                              f'{results["gt_ifbidirection_map"].shape}')
                data = to_tensor(results['gt_ifbidirection_map'].astype(np.int64))
            gt_ifbidirection_data = dict(data=data)
            data_sample.set_data(dict(gt_ifbidirection_map=PixelData(**gt_ifbidirection_data)))

        if 'gt_ifboundary_map' in results:
            if len(results['gt_ifboundary_map'].shape) == 2:
                data = to_tensor(results['gt_ifboundary_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'ifboundary map, usually the ifboundary '
                              'map is 2D, but got '
                              f'{results["gt_ifboundary_map"].shape}')
                data = to_tensor(results['gt_ifboundary_map'].astype(np.int64))
            gt_ifboundary_data = dict(data=data)
            data_sample.set_data(dict(gt_ifboundary_map=PixelData(**gt_ifboundary_data)))
        
        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        if 'gt_instance_map' in results:
            gt_instance_data = dict(
                data=to_tensor(results['gt_instance_map'][None,
                                                         ...].astype(np.int64)))
            data_sample.set_data(
                dict(gt_instance_map=PixelData(**gt_instance_data)))

        if 'direction_map' in results:
            flip_direction = results.get('flip_direction', None)
            direction_png = results['direction_map']
            direction_x = direction_png[..., 1] # the order should be the same as the generation file
            direction_y = direction_png[..., 0]
            
            # convert x,y png data to angle data
            direction_x = (direction_x / 127.5) - 1
            direction_y = (direction_y / 127.5) - 1
            direction_angle = np.arctan2(direction_y, direction_x)
            # consider the flip direction
            if flip_direction == 'horizontal':
                direction_angle = np.pi - direction_angle # 0 ~ 2pi
                # turn the angle to [-pi, pi], bidirectional
                direction_angle = np.where(direction_angle > np.pi, direction_angle - 2 * np.pi, direction_angle)
            elif flip_direction == 'vertical':
                direction_angle = - direction_angle # -pi ~ pi
            elif flip_direction == 'diagonal':
                direction_angle = np.pi + direction_angle # 0 ~ 2pi
                # turn the angle to [-pi, pi], bidirectional
                direction_angle = np.where(direction_angle > np.pi, direction_angle - 2 * np.pi, direction_angle)
            elif flip_direction == None:
                pass
            else:
                raise ValueError(f'Unsupported flip_direction {flip_direction}')

            # transform the angle to [0, pi], unidirectional
            # direction_angle = np.where(direction_angle < 0, direction_angle + np.pi, direction_angle)
            # direction_angle = np.clip(direction_angle, 0, np.pi)

            # use the angle to [-pi, pi]
            direction_angle = np.clip(direction_angle, -np.pi, np.pi) # shape: (H, W)
            # NOTE:we use the unit vector in the loss part not here
            direction_data = dict(
                data=to_tensor(direction_angle[None, ...].astype(np.float32))) # adding None to add a new axis for batch size
            
            data_sample.set_data(
                dict(direction_map=PixelData(**direction_data)))
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
