[TOC]

---

# 代码解读和记录

## config

```
# model settings
model = dict(
    type='FasterRCNNHBBOBB',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
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
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=17,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    rbbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    rbbox_head=dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=17,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        reg_class_agnostic=False,
        with_module=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,	# anchor超出图像边界的可容忍像素范围，-1代表不提出越界anchor
        pos_weight=-1,
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
            type='MaxIoUAssignerCy',
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
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        # score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=1000),
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=2000),
    rrcnn=dict(
        # score_thr=0.05, nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img=1000)
        score_thr = 0.05, nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img = 2000)
# soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'DOTA1_5Dataset_v2'
data_root = 'data/dota1_1024/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA1_5_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA1_5_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test1024/DOTA1_5_test1024.json',
        img_prefix=data_root + 'test1024/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=12)
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
work_dir = './work_dirs/faster_rcnn_h-obb_r50_fpn_1x_dota1_5'
load_from = None
resume_from = None
workflow = [('train', 1)]

```


## anchor setting

head中的anchor设置参数为：

```
octave_base_scale=4,
scales_per_octave=3,
anchor_ratios=[0.5, 1.0, 2.0],
anchor_strides=[8, 16, 32, 64, 128],
```

其中`anchor_strides`代表的是anchor感受野，降采样步长，`scales_per_octave`控制尺度，`octave_base_scale`是anchor面积的开方sqrt(wh)，和ratio一起生成anchor。**每个位置的anchor数目是scales_per_octave*anchor_ratios**。具体而言，例如stride为32时，生成九个anchor为：

```
(32*8*（2^(0/3)）/sqrt(2), 32*8（2^(0/3)）* sqrt(2)), 
(32*8*（2^(1/3)）/sqrt(2), 32*8（2^(1/3)）* sqrt(2)),
(32*8*（2^(2/3)）/sqrt(2), 32*8（2^(2/3)）* sqrt(2));
(32*8*（2^(0/3)）/sqrt(1), 32*8（2^(0/3)）* sqrt(1)),
(32*8*（2^(1/3)）/sqrt(1), 32*8（2^(1/3)）* sqrt(1)),
(32*8*（2^(2/3)）/sqrt(1), 32*8（2^(2/3)）* sqrt(1));
(32*8*（2^(0/3)）/sqrt(0.5), 32*8（2^(0/3)）* sqrt(0.5)), 
(32*8*（2^(1/3)）/sqrt(0.5), 32*8（2^(1/3)）* sqrt(0.5)),
(32*8*（2^(2/3)）/sqrt(0.5), 32*8（2^(2/3)）* sqrt(0.5)).
```

很明白了吧，scales_per_octave的个数代表`2^0`开始的多少个尺度，等分1作为幂指数；anchor_ratios和scales_per_octave其实推算一下不难得出：$h= s\sqrt{r} , w= s/ \sqrt{r} $，其中s就是octave_base_scale，r就是ratio了，wh顺序无所谓，反正对称的。

## mmdet

### core

#### anchor

* anchor_target_rbbox

  `anchor_target_rbbox`是调用接口，主体实现是`anchor_target_rbbox_single`函数，其中首先`inside_flags`通过调用`anchor_inside_flags`选出边界约束的有效anchor，即cfg文件的allowed_border参数设置的是anchor超出图像的可容忍边界，然后根据是否进行sampling采样（focal loss和GHM由于全员加权，不采样）。在根据要求分配正负样本，以及bbox_encode成delta的形式（这里涉及一个mask转到obb的过程，程序很多地方都这么转来转去的，估计是为了统一分割和检测的codebase吧），同时给每个anchor计算了一个label weight后面用作loss的加权系数。
  
* anchor_generator

  直接返回anchor，但是注意他是shape (num_all_anchors, 4)，将anchor全部叠加到一个维度了，格式是xyxy，方向为：先固定cxcy排列单pix的anchor，然后滑动cx固定cy，最后滑动cy

#### bbox

##### assigners

###### max_iou_assigner_hbb_cy

计算**水平**anchor之间的IoU并进行正负样本选取分配，如果是rbox，只当做水平的（此处AerialDetection的anchor生成函数由于本身就没有角度，用这个就行，后面如果加角度anchor此处要改）;

pos_iou_thr和neg_iou_thr分别是正负样本阈值，min_pos_iou是maxiou补偿选出的那个anchor必须大于这个thres，否则宁可不要;

assign( )是调用接口，bboxes为anchor：(num_anchors, 4)，gt_bboxes：(num_gt, ,4)，gt_labels：(num_gt)。计算出iou后传入assign_wrt_overlaps( )实现; 

##### samplers

### model
#### anchor_heads

anchor的前向传播head第一层封装会来这里，例如RetinaNet的RetinaHeadRbbox，然后调用父类方法。

##### anchor_head_rbbox

只有一个类AnchorHeadRbbox；

**调用路径**：forward比较简单，两个branch每个一组卷积输出（reg多回归一个角度）；重点关注怎么计算loss的：loss方法是调用接口，外层通过对head的这个方法进行接口调用，重写新的head需要定义这个，例如训练时建立RetinaNet模型时会向上继承其父类SingleStageDetectorRbbox的forward_train方法，而这里定义了self.bbox_head.loss，由于bbox_head已经在之前build为RetinaHeadRbbox了，所以会进一步调用他的父类AnchorHeadRbbox的loss方法，就到了本文件了。

**loss计算**：首先经由anchor_target_rbbox将每张图像的anchor list拿到编码成delta的分配后的正负样本划分结果以及加权系数，然后调用loss_single结合加权系数算loss就行了



---

# 一些文件记录

##### result.pkl

存储的是检测结果，用于算mAP的，所以这里面的dets特点是经过NMS，但是conf未筛选，如果想从这里拿到最终检测结果加个conf_thres就可以了（但没必要，有现成的api）。

格式是：

* 第一层list：长度为num_pics，检测图片的个数；
* 第二层list：长度为类别数，因为是DOTA格式的eval，一类一个文件的存储方式，顺序和classname定义一致，可通过`dataset.CLASSES`读取；
* 第三层numpy数组：shape为(num_dets, 9)，9=8+conf

读取方法是`mmcv.load(resultfile)`，加载得到的结果为所有图片list，每个图片有十个子list，存储了该类目标的检测结果个数。

---



# 问题合集

### 历史版本问题

---

##### RuntimeError: any only supports torch.uint8 dtype

数据类型问题，可以参考[这里](https://github.com/open-mmlab/mmdetection/pull/2939/commits/ccc9fe69a37ddfbce48f37ac9c4dc56346d6589d)，将`mmdet/core/anchor/utils.py`作如下改动：

```
return inside_flags
改为
return inside_flags.to(torch.bool)
```

---

##### assert len(indices) == self.num_samples

出现原因是图片数除以img_per_GPU不得整数，参考[这里](https://github.com/open-mmlab/mmdetection/pull/1610/commits/1e1297897b46f094db13e15849a8cfd0942bcb6d)，将`mmdet/datasets/loader/sampler.py`的如下代码段进行修改即可：

```
indice = np.concatenate([indice, indice[:num_extra]])
改为；
indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
```

---

### 安装问题

首先，安装不要install，develop才能自由改动

```
python setup.py develop
```

##### cuda相关的问题

类似这种：

1. mmdet/ops/utils/src/compiling_info.cpp:6:10: fatal error: cuda_runtime_api.h: **No such file or directory**
2. ImportError: **cannot import name 'deform_conv_cuda'**
3. ImportError: libcudart.so.10.1: cannot open shared object file: **No such file or directory**

等问题，出现原因是编译mmdetection的CUDA版本问题，因为你编译的时候默认指定的cuda路径是/usr/local的cuda目录，而这个是软连接，连接到同目录下特定版本的cuda，所以直接改一下路径就行（或者你可以自己建立新的软连接，总之保持编译和运行版本一样就行）：

```
export CUDA_HOME=/usr/local/cuda-10.0    # 其中cuda版本根据自己的进行调整
source ~/.bashrc
```



---

##### No module named 'mmcv.cnn.weight_init'

这个以及其他的mmcv问题可能都是版本，自动装的mmcv会很高，安装合适的版本即可，一般是降版本，我用的如：

```
pip install mmcv==0.4.3
```

##### undefined symbol: _ZN6caffe26detail37_typeMetaDataInstance_preallocated_32E

这是个很神奇的问题，网上找了很久没有正确答案，很多人重装偶尔可行，有的人就是不行，只好弃掉自己的代码，但真相居然是：在import cuda函数之前先import torch即可！

所以写程序注意：1. 不懂就别瞎改 2. 多用git记录代码版本

##### 其他问题

