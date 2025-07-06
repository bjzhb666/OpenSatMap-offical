# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from typing import List, Tuple, Any
from torch import Tensor
import math
from einops import rearrange
import warnings
from ..losses import accuracy

class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        # torch.manual_seed(0)
        # bases = torch.randn((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


class UpsampleNetwork2(nn.Module):
    """
    upsample network, used to upsample the feature map from (N, 480, 256, 256) to (N, L, 2048, 2048)
    L is AE embedding dimension
    """
    def __init__(self, L, in_channels, upsample_channels=1024):
        super(UpsampleNetwork2, self).__init__()
        # 设定Feedforward Network(FFN)的结构，这里简单使用Conv2d作为演示
        self.ffn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=upsample_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(upsample_channels/64), out_channels=int(upsample_channels/64), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(upsample_channels/64), out_channels=int(upsample_channels/64), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.ffn2 = nn.Sequential(
            nn.Conv2d(in_channels=int(upsample_channels/64), out_channels=L, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
        )
    
    def forward(self, x):
        # 输入bs, 480, 256, 256
        bs = x.size(0)
        x = self.ffn1(x) # 变成bs, 1024, 256, 256
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (p1 h) (p2 w)', p1=8, p2=8) # 变成bs, 16, 2048, 2048
        x = self.conv1(x) # 经过Conv1到bs, 16, 2048, 2048
        x = self.conv2(x) # 经过Conv2到bs, 16, 2048, 2048
        x = self.ffn2(x) # 经过FFN到bs, L, 2048, 2048
        
        return x


class AdditionalSemHead(nn.Module):
    def __init__(self, in_channels, number_classes, conv_cfg, norm_cfg, act_cfg, 
                 ham_channels, channels, ham_kwargs, dropout_ratio, out_channels=None, **kwargs):
        super(AdditionalSemHead, self).__init__()
        
        if out_channels is None:
            if number_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert `seg_logits` into a prediction'
                              'applying a threshold')
            out_channels = number_classes

        if out_channels != number_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={number_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        
        self.num_classes = number_classes
        self.out_channels = out_channels
        
        self.squeeze = ConvModule(
            sum(in_channels),
            ham_channels, 1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            ham_channels,
            channels, 1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
    
    def forward(self, inputs):
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)
        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, embed_dims: List[int]):
        super(UpsampleDecoder, self).__init__()
        
        new_input_channel1=in_channels + embed_dims[1]
        self.deconv1 = nn.ConvTranspose2d(new_input_channel1, new_input_channel1//2, kernel_size=3, stride=2, 
                                          padding=1, output_padding=1)
        self.conv1 = DoubleConv(new_input_channel1//2, new_input_channel1//2)
        
        new_input_channel2 = new_input_channel1//2 + embed_dims[0] 
        self.deconv2 = nn.ConvTranspose2d(new_input_channel2, new_input_channel2//2, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)
        self.conv2 = DoubleConv(new_input_channel2//2, new_input_channel2//2)
        # self.deconv3 = nn.ConvTranspose2d(new_input_channel2//2, new_input_channel2//4, kernel_size=3, stride=2,
        #                                     padding=1, output_padding=1)
        # self.conv3 = DoubleConv(new_input_channel2//4, new_input_channel2//4)
        
    def forward(self, x, ori_input):
        x = torch.cat([x, ori_input[1]], dim=1)
        x = self.deconv1(x)
        x = self.conv1(x)
        x = torch.cat([x, ori_input[0]], dim=1)
        x = self.deconv2(x)
        x = self.conv2(x)
        # x = self.deconv3(x)
        # x = self.conv3(x)
        
        return x


@MODELS.register_module()
class LightHamInstanceHead(BaseDecodeHead):
    """SegNeXt decode head.

    This decode head is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Specifically, LightHamHead is inspired by HamNet from
    `Is Attention Better Than Matrix Decomposition?
    <https://arxiv.org/abs/2109.04553>`.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, AE_dimension:int, has_AE_head:bool, has_direction_head: bool, has_line_type_head: bool,
                     ham_channels=512, loss_instance_decode=dict(type='AELoss',
                     loss_weight=1.0, push_loss_factor=1.0, minimum_instance_pixels=0),
                     loss_direction_decode=dict(type='MSERegressionLoss', loss_weight=1.0),
                     loss_linenum_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     loss_linetype_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     ham_kwargs=dict(), num_color_classes=5, num_line_types=11, num_linenums = 5, num_attributes = 9,
                     num_bidirections = 3, num_boundary_types = 3, angle_class=360, embed_dims=[64, 128, 320, 512],
                     seg_upsample_channel=1024, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels
        self.seg_upsample_channel = seg_upsample_channel
        num_concat_channel = embed_dims[1]//2+embed_dims[0]

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align1 = ConvModule(
            self.ham_channels,
            self.seg_upsample_channel,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.AE_dimension = AE_dimension
        
        self.num_color_classes = num_color_classes
        self.num_line_types = num_line_types
        self.num_linenums = num_linenums
        self.num_attributes = num_attributes
        self.num_bidirections = num_bidirections
        self.num_boundary_types = num_boundary_types
        
        self.has_AE_head = has_AE_head
        self.has_direction_head = has_direction_head
        self.has_line_type_head = has_line_type_head

        if self.has_line_type_head:
            # define additional seg head, solid and dashed line ...
            self.line_type_seg_head = AdditionalSemHead(
                number_classes=self.num_line_types,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                ham_channels=ham_channels,
                ham_kwargs=ham_kwargs,
                **kwargs
            )
            # define loss
            if isinstance(loss_linetype_decode, dict):
                self.loss_linetype_decode = MODELS.build(loss_linetype_decode)
            elif isinstance(loss_linetype_decode, (list, tuple)):
                self.loss_linetype_decode = nn.ModuleList()
                for loss_linetype in loss_linetype_decode:
                    self.loss_linetype_decode.append(MODELS.build(loss_linetype))
            else:
                raise TypeError(f'loss_linetype_decode must be a dict or sequence of dict,\
                    but got {type(loss_linetype_decode)}')

        if self.has_direction_head:
            # direction head initialization
            self.direction_head = UpsampleNetwork2(L=1, in_channels=sum(self.in_channels), upsample_channels=1024)
            # loss direction decode (regression loss)
            if isinstance(loss_direction_decode, dict):
                self.loss_direct_decode = MODELS.build(loss_direction_decode)
            elif isinstance(loss_direction_decode, (list, tuple)):
                self.loss_direct_decode = nn.ModuleList()
                for loss_direct in loss_direction_decode:
                    self.loss_direct_decode.append(MODELS.build(loss_direct))
            else:
                raise TypeError(f'loss_direct_decode must be a dict or sequence of dict,\
                    but got {type(loss_direction_decode)}')
        
        # Upsample network for segmentation
        self.seg_upsample = UpsampleDecoder(in_channels=self.seg_upsample_channel, 
                                            embed_dims=embed_dims)
        self.align2 = ConvModule(
            self.seg_upsample_channel//4+num_concat_channel//2,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        
    def forward(self, inputs_ori):
        """Forward function.
        inputs: list of feature maps from different levels (list of Tensor):
            - inputs[0]: the feature map from the highest level torch.Size([bs, 32, 512, 512])
            - inputs[1]: the feature map from the second highest level torch.Size([bs, 64, 256, 256])
            - inputs[2]: the feature map from the third highest level torch.Size([bs, 160, 128, 128])
            - inputs[3]: the feature map from the fourth highest level torch.Size([bs, 256, 64, 64])
        Returns:
            output: the output feature map torch.Size([bs, num_cls, 256, 256])
            tag_map_2048: the output tag map torch.Size([bs, 1, 2048, 2048])
            direction_map_2048: the output direction map torch.Size([bs, 1, 2048, 2048])
        """

        inputs = self._transform_inputs(inputs_ori) # choose the last three layers

        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ] # resize all feature maps to the same size: 256*256

        inputs = torch.cat(inputs, dim=1) # bs, 480 (960), 256, 256
        
        # tag head: apply a conv block to squeeze feature map
        # First step: resize the feature map to bs,1,256,256 (get tag map)
        # tag_map_256 = self.tag(inputs)
        # Second step: upsample the tag map to bs,1,2048,2048
        # tag_map_2048 = self.upsample(inputs)

        # seg head: apply a conv block to squeeze feature map
        x = self.squeeze(inputs) # x shape: bs, 256, 256, 256
        # apply hamburger module
        x = self.hamburger(x) # bs, 256, 256, 256
        # apply a conv block to align feature map
        output = self.align1(x) # output shape: bs, 1024, 256, 256
        if self.dropout_ratio > 0:
            output = self.dropout(output)
        output = self.seg_upsample(output, inputs_ori) # output shape: bs, 256, 2048, 2048
        output = self.align2(output) # output shape: bs, 256, 1024, 1024
        output = self.cls_seg(output) # output shape: bs, num_cls, 2048, 2048
        # line type seg head
        if self.has_line_type_head:
            line_type_seg = self.line_type_seg_head(inputs)
        else:
            line_type_seg = None

        # direction_head: predict a direction for each pixel
        if self.has_direction_head:
            direction_seg = self.direction_head(inputs)
            direction_seg = torch.clamp(direction_seg, -torch.pi, torch.pi)
        else:
            direction_seg = None
        
        return output, None, direction_seg, line_type_seg, None
    
    def _get_ignore_map(self, batch_data_samples: SampleList, ignore_index: int) -> Tensor:

        ignore_map = [
            data_sample.gt_sem_seg.data==ignore_index for data_sample in batch_data_samples
        ]
        return torch.stack(ignore_map, dim=0)
    
    def _stack_batch_instance_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_instance_segs = [
            data_sample.gt_instance_map.data for data_sample in batch_data_samples
        ] # list: len=batch_size, each element is a tensor of shape (1, 2048, 2048)
        
        return torch.stack(gt_instance_segs, dim=0)
    
    def loss_instance_by_feat(self, tag_map_2048: Tensor, batch_data_samples: SampleList) -> dict:
        """Compute AE loss for instance segmentation.

        Args:
            tag_map_2048 (Tensor): The output tag map. Shape: (batch_size, 1, 2048, 2048)
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg` and `gt_instance_map`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_instance = self._stack_batch_instance_gt(batch_data_samples) # shape: (batch_size, 1, 2048, 2048)
        ignore_map = self._get_ignore_map(batch_data_samples, self.ignore_index) # shape: (batch_size, 1, 2048, 2048)
        loss_instance = dict()
        
        if not isinstance(self.loss_instance_decode, nn.ModuleList):
            losses_instance_decode = [self.loss_instance_decode]
        else:
            losses_instance_decode = self.loss_instance_decode
        for loss_instance_decode in losses_instance_decode:
            if loss_instance_decode.loss_name not in loss_instance:
                loss_instance[loss_instance_decode.loss_name] = loss_instance_decode(
                    tag_map_2048, gt_instance, ignore_position = ignore_map)
            else:
                loss_instance[loss_instance_decode.loss_name] += loss_instance_decode(
                    tag_map_2048, gt_instance, ignore_position = ignore_map)
        
        if 'loss_ae' in loss_instance:
            loss_instance_pull_push = {}
            loss_instance_pull_push['loss_ae_pull'] = loss_instance['loss_ae'][0]
            loss_instance_pull_push['loss_ae_push'] = loss_instance['loss_ae'][1]       
        elif 'loss_moco' in loss_instance:
            loss_instance_pull_push = {}
            loss_instance_pull_push['loss_moco'] = loss_instance['loss_moco'][0]
            loss_instance_pull_push['acc_moco'] = torch.tensor([loss_instance['loss_moco'][1]]).to(tag_map_2048.device)

        return loss_instance_pull_push

    def _stack_batch_direct_gt(self, batch_data_samples: SampleList) -> Tensor:
        
        gt_directions = [
            data_sample.direction_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_directions, dim=0)

    def loss_direct_by_feat(self, direct_map: Tensor, batch_data_samples: SampleList) -> dict:
        """Compute regression loss for direction prediction.

        Args:
            direct_map_2048 (Tensor): The output direction map. Shape: (batch_size, 1, 2048, 2048)
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg` and `gt_instance_map`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        gt_direction = self._stack_batch_direct_gt(batch_data_samples)
        seg_label = self._stack_batch_gt(batch_data_samples)
        front_map = (seg_label != 0) & (seg_label != self.ignore_index)
        
        loss_direct = dict()
        if not isinstance(self.loss_direct_decode, nn.ModuleList):
            losses_direct_decode = [self.loss_direct_decode]
        else:
            losses_direct_decode = self.loss_direct_decode

        for loss_direct_decode in losses_direct_decode:
            if loss_direct_decode.loss_name not in loss_direct:
                loss_direct[loss_direct_decode.loss_name], loss_direct['angle_mean_diff'] = loss_direct_decode(
                    direct_map, gt_direction, front_map)
            else:
                loss_value, angle_mean_diff = loss_direct_decode(
                    direct_map, gt_direction, front_map)
                loss_direct[loss_direct_decode.loss_name] += loss_value
                # loss_direct['angle_mean_diff'] += angle_mean_diff 
        return loss_direct

    def _stack_batch_line_type_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_line_type = [
            data_sample.gt_line_type_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_line_type, dim=0)
    
    def loss_linetype_by_feat(self, line_type_seg_logits: Tensor, batch_data_samples: SampleList) -> dict:
        """Compute loss for line type segmentation.

        Args:
            line_type_seg_logits (Tensor): The output line type segmentation logits. Shape: (batch_size, num_line_types, 512, 512)
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg` and `gt_instance_map`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_line_type_label = self._stack_batch_line_type_gt(batch_data_samples)
        loss = dict()
        line_type_logits = resize(
            input=line_type_seg_logits,
            size = gt_line_type_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        if self.sampler is not None:
            line_type_weight = self.sampler.sample(line_type_logits, gt_line_type_label)
        else:
            line_type_weight = None
        gt_line_type_label = gt_line_type_label.squeeze(1)

        if not isinstance(self.loss_linetype_decode, nn.ModuleList):
            losses_linetype_decode = [self.loss_linetype_decode]
        else:
            losses_linetype_decode = self.loss_linetype_decode

        for loss_linetype_decode in losses_linetype_decode:
            if loss_linetype_decode.loss_name not in loss:
                loss[loss_linetype_decode.loss_name+'_linetype'] = loss_linetype_decode(
                    line_type_logits, gt_line_type_label, 
                    weight=line_type_weight, ignore_index=self.ignore_index)
            else:
                loss[loss_linetype_decode.loss_name+'_linetype'] += loss_linetype_decode(
                    line_type_logits, gt_line_type_label, 
                    weight=line_type_weight, ignore_index=self.ignore_index)
                
        loss['acc_linetype'] = accuracy(
            line_type_logits, gt_line_type_label, ignore_index=self.ignore_index)
        return loss
                
    def _stack_batch_linenum_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_linenum = [
            data_sample.gt_line_num_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_linenum, dim=0)
    
    def loss_linenum_by_feat(self, linenum_seg_logits: Tensor, batch_data_samples: SampleList) -> dict:
        """Compute loss for line number segmentation.

        Args:
            linenum_seg_logits (Tensor): The output line number segmentation logits. Shape: (batch_size, num_linenums, 512, 512)
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg` and `gt_instance_map`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_linenum_label = self._stack_batch_linenum_gt(batch_data_samples)
        loss = dict()
        linenum_logits = resize(
            input=linenum_seg_logits,
            size=gt_linenum_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        if self.sampler is not None:
            linenum_weight = self.sampler.sample(linenum_logits, gt_linenum_label)
        else:
            linenum_weight = None
        gt_linenum_label = gt_linenum_label.squeeze(1)

        if not isinstance(self.loss_linenum_decode, nn.ModuleList):
            losses_linenum_decode = [self.loss_linenum_decode]
        else:
            losses_linenum_decode = self.loss_linenum_decode

        for loss_linenum_decode in losses_linenum_decode:
            if loss_linenum_decode.loss_name not in loss:
                loss[loss_linenum_decode.loss_name+'_linenum'] = loss_linenum_decode(
                    linenum_logits, gt_linenum_label, 
                    weight=linenum_weight, ignore_index=self.ignore_index)
            else:
                loss[loss_linenum_decode.loss_name+'_linenum'] += loss_linenum_decode(
                    linenum_logits, gt_linenum_label, 
                    weight=linenum_weight, ignore_index=self.ignore_index)
                
        loss['acc_linenum'] = accuracy(
            linenum_logits, gt_linenum_label, ignore_index=self.ignore_index)
        return loss

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, tag_map_2048, direction_seg, line_type_seg_logits, \
            linenum_seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        # import pdb; pdb.set_trace()
        if tag_map_2048 is not None:
            losses_instance = self.loss_instance_by_feat(tag_map_2048, batch_data_samples)
            losses.update(losses_instance)
        if direction_seg is not None:
        # update losses dict with instance loss
            losses_direct = self.loss_direct_by_feat(direction_seg, batch_data_samples)
            losses.update(losses_direct)
        if line_type_seg_logits is not None:
            losses_linetype = self.loss_linetype_by_feat(line_type_seg_logits, batch_data_samples)
            losses.update(losses_linetype)
        if linenum_seg_logits is not None:
            losses_linenum = self.loss_linenum_by_feat(linenum_seg_logits, batch_data_samples)
            losses.update(losses_linenum)
       
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, tag_map_2048, direct_map_2048, line_type_seg_logits, \
            linenum_seg_logits = self.forward(inputs)
        if linenum_seg_logits is not None:
            linenum_seg_logits = self.predict_by_feat(linenum_seg_logits, batch_img_metas)
        if line_type_seg_logits is not None:
            line_type_seg_logits = self.predict_by_feat(line_type_seg_logits, batch_img_metas)
        
        return self.predict_by_feat(seg_logits, batch_img_metas), tag_map_2048, direct_map_2048, \
           line_type_seg_logits, linenum_seg_logits
