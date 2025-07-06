dataset_type = 'SatelliteSegDataset'
# data_root = 'data/satellite/'
data_root = 'data/satellite_seg_512/'
crop_size = (2048, 2048) # random crop size
bezier_order = 3
dilate_kernel = 4

train_pipeline = [
    dict(type='LoadImageFromFile'), # 第1个流程，从文件路径里加载图像
    dict(type='LoadAnnotations'),  # 第2个流程，对于当前图像，加载它的标注图像
    # dict(type='RandomResize',  # 调整输入图像大小(resize)和其标注图像的数据增广流程
    #     scale=(2048, 1024),  # 图像裁剪的大小
    #     ratio_range=(0.5, 2.0),  # 数据增广的比例范围
    #     keep_ratio=True),  # 调整图像大小时是否保持纵横比
    # dict(type='RandomCrop',  # 随机裁剪当前图像和其标注图像的数据增广流程
    #     crop_size=crop_size,
    #     cat_max_ratio=0.95),  # The maximum ratio that single category could occupy.
    dict(type='Resize',  # 调整输入图像大小(resize)和其标注图像的数据增广流程
    scale=crop_size),  # 图像裁剪的大小
    dict(type='DilateGT', kernel_size=dilate_kernel),  # 对标注图像进行膨胀操作    
    dict(type='RandomFlip',  # 翻转图像和其标注图像的数据增广流程
        prob=0.5),  # 翻转图像的概率
    # dict(type='NormalizeLineCoordinate', img_shape=crop_size),  # 标准化线的坐标
    dict(type='PhotoMetricDistortion'),  # 光学上使用一些方法扭曲当前图像和其标注图像的数据增广流程，和GT无关，不用改
    dict(type='PackSegInputs')  # 打包用于语义分割的输入数据
]

test_pipeline = [
    dict(type='LoadImageFromFile'),  # 第1个流程，从文件路径里加载图像
    dict(type='Resize',  # 调整输入图像大小(resize)和其标注图像的数据增广流程
    scale=crop_size),  # 图像裁剪的大小
    # 在' Resize '之后添加标注图像
    # 不需要做调整图像大小(resize)的数据变换  
    dict(type='LoadAnnotations'),  # 加载数据集提供的语义分割标注
    dict(type='DilateGT', kernel_size=dilate_kernel),  # 对标注图像进行膨胀操作   
    dict(type='PackSegInputs')  # 打包用于语义分割的输入数据
]

train_dataloader = dict(  # 训练数据加载器(dataloader)的配置
    batch_size=2,  # 每一个GPU的batch size大小
    num_workers=2,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 训练时进行随机洗牌(shuffle)
    dataset=dict(  # 训练数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train_palette_see'),  # 训练数据的前缀
        # json_path='anno_json/lines_train2017.json',
        # json_path='anno_json/line_train_3rd.json',
        # bezier_order=bezier_order,  # 贝塞尔曲线的阶数
        # cache=True,
        # cache_path='data/satellite/cache/train_cache_3rd.pkl',
        pipeline=train_pipeline)) # 数据处理流程，它通过之前创建的train_pipeline传递。

val_dataloader = dict(
    batch_size=1,  # 每一个GPU的batch size大小
    num_workers=4,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='DefaultSampler', shuffle=False),  # 训练时不进行随机洗牌(shuffle)
    dataset=dict(  # 测试数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val_palette_see'),  # 测试数据的前缀
        # json_path='anno_json/lines_val2017.json',
        # json_path='anno_json/line_val_3rd.json',
        # bezier_order=bezier_order,  # 贝塞尔曲线的阶数
        # cache=True,
        # cache_path='data/satellite/cache/val_cache_3rd.pkl',
        pipeline=test_pipeline))  # 数据处理流程，它通过之前创建的test_pipeline传递。

test_dataloader = val_dataloader

# 精度评估方法，我们在这里使用 IoUMetric 进行评估
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice', 'mFscore'])
test_evaluator = val_evaluator