# model settings
model = dict(
    type='FasterRCNN',
    # 注意这个地方： inference时，demo程序需要覆盖为None；training 时，如果对网络有更改，这个模型是用不了的，也要写None重新自己训练
    pretrained='modelzoo://resnet50',   
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,               # stage特征输出的层级
        out_indices=(0, 1, 2, 3),   # 对应的上面stage1-4前向传播输出结果的索引
        frozen_stages=1,            # 冻结的stage数量，即该stage不更新参数（1-4），-1表示所有的stage都更新参数
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048], # 输入的各个stage的通道数,都是stage的最后一层
        out_channels=256,                   # FPN输出的特征层的通道数
        num_outs=5),                        # FPN输出的特征层的数量（有一个直接的降采样所以多了一层）
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,                    # RPN网络的输入通道数
        feat_channels=256,                  # 特征层的通道数
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,          # 全连接层数量
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,           # 是否采用class_agnostic的方式来回归
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',  # RPN网络的正负样本划分
            pos_iou_thr=0.7,        # 正样本的iou阈值
            neg_iou_thr=0.3,        # 负样本iou
            min_pos_iou=0.3,        # 正样本的iou最小值。如果分配给gt的anchors中最大IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),     # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',   # 正负样本提取器类型
            num=256,                # 需提取的正负样本数量  
            pos_fraction=0.5,       # 正样本比例
            neg_pos_ub=-1,          # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=False), # 把ground truth加入proposal作为正样本
        allowed_border=0,           # 允许在bbox周围外扩一定的像素
        pos_weight=-1,              # 正样本权重，-1表示不改变原始的权重
        debug=False),   
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(                    # 模型搭建不会使用，forward时会调用进行设置（train时传入了train和test两个cfg配置）。推断时的RPN参数
    rpn=dict(
        nms_across_levels=False,    # 在所有的fpn层内做nms
        nms_pre=1000,               # 在nms之前保留的得分最高的proposal数量
        nms_post=1000,              # 在nms之后保留的得分最高的proposal数量
        max_num=1000,               # 在后处理完成之后保留的proposal数量
        nms_thr=0.7,
        min_bbox_size=0),           # 最小bbox尺寸
    rcnn=dict(  
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)   # max_per_img为最终输出的det bbox数量上限
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,     # 对图像进行resize时的最小单位，32表示所有的图像都会被resize成32的倍数
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(               # inference阶段的配置
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]       # 工作流程，代表train一轮。更详细的解释见runner注释，这里就不展开，一般只用这个就行
