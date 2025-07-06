_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py',
    '../_base_/datasets/satellite_seg_instance.py'
]
AE_dimension=16
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (2048, 2048)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True ,
    pad_val=0, # 如果要padding原图，用0
    seg_pad_val=255, # 如果要padding mask标签，用255
    size=crop_size,
    test_cfg=dict(size_divisor=32)
    ) # 先在train pipeline处理数据，然后再过data preprocesser
# data_preprocessor = dict(type='SegDataPreProcessor',
#                          size = crop_size,
#                          mean=[123.675, 116.28, 103.53],
#                          std=[58.395, 57.12, 57.375])
tag_dict = {'Gradual': dict(type='GradualReduction', output_channel=AE_dimension),
            'Direct': dict(type='DirectReduction', output_channel=AE_dimension),
            'SEBlock': dict(type='SEBlock',input_channels=480, output_channel=AE_dimension, reduction=16)}
direct_dict = {
    'Gradual': dict(type='GradualReduction', output_channel=1),
    'Direct': dict(type='DirectReduction', output_channel=1),
    'SEBlock': dict(type='SEBlock',input_channels=480, output_channel=1, reduction=16)
}
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    has_AE_head=True,
    has_direction_head=True,
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamInstanceHead',
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        ignore_index=100,
        # tag_type=tag_dict['Gradual'], # feature map转为tag的方式
        # direction_type = direct_dict['Gradual'], # feature map转为direction的方式
        AE_dimension = 16,
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3], # 对应backbone的stage，从0开始，这里是第2，第3，第4个stage（后三层）
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=4, # 分割前景的种类数目。 通常情况下，cityscapes 为19，VOC为21，ADE20k 为150
        num_color_classes=5,
        num_line_types=11,
        num_linenums = 5,
        num_attributes = 9,
        num_bidirections = 3,
        num_boundary_types = 3,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=[dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0, class_weight=[1, 20, 30, 40], avg_non_ignore=True),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
        # loss_instance_decode=dict(
        #     type='AELoss', loss_weight=1, push_loss_factor=1, minimum_instance_pixels=1),
        loss_instance_decode=dict(
            type='MocoLoss', loss_weight=0.0, minimum_instance_pixels=1),
        loss_direction_decode=dict(
            type='MSERegressionLoss', loss_weight=0.0),
        loss_linenum_decode=[dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1, 20, 50, 20, 30], avg_non_ignore=True),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
        loss_linetype_decode=[dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0, class_weight=[1, 50, 40, 40, 40, 40, 40, 40, 40, 50, 50], avg_non_ignore=True),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)],
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
train_dataloader = dict(batch_size=1)

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00008, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head.seg_head': dict(lr_mult=10.),
            'head.tag_head': dict(lr_mult=10.),
            'head.direction_head': dict(lr_mult=10.),
            'head.line_type_seg_head': dict(lr_mult=10.),
            'head.line_num_seg_head': dict(lr_mult=10.),
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=2e-6, by_epoch=False, begin=0, end=400),
    dict(
        type='PolyLR',
        power=1.0,
        begin=400,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# 精度评估方法，我们在这里使用 InstanceIoUMetric 进行评估
val_evaluator = dict(type='InstanceIoUMetric', iou_metrics=['mIoU','mDice', 'mFscore'], 
                     ignore_index=100, save_ori_prediction=False, use_seg_GT=False, GT_without_Water=False, minimal_area = 1)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend'),
                 dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=0.7)