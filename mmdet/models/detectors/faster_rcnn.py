from .two_stage import TwoStageDetector
from ..registry import DETECTORS

import ipdb

# 利用super的构造函数继承，递归地搭建了计算图！

@DETECTORS.register_module  
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)


'''
训练时传入这里的参数是：
self:Faster RCNN的组建
backbone = {'type': 'ResNet', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'style': 'pytorch'}
rpn_head = {'type': 'RPNHead', 'in_channels': 256, 'feat_channels': 256, 'anchor_scales': [8], 'anchor_ratios': [0.5, 1.0, 2.0], 'anchor_strides': [4, 8, 16, 32, 64], 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [1.0, 1.0, 1.0, 1.0], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 0.1111111111111111, 'loss_weight': 1.0}}
bbox_roi_extractor = {'type': 'SingleRoIExtractor', 'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 2}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}
bbox_head = {'type': 'SharedFCBBoxHead', 'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 81, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.1, 0.1, 0.2, 0.2], 'reg_class_agnostic': False, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 1.0, 'loss_weight': 1.0}}
train_cfg = {'rpn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.7, 'neg_iou_thr': 0.3, 'min_pos_iou': 0.3, 'ignore_iof_thr': -1}, 'sampler': {'type': 'RandomSampler', 'num': 256, 'pos_fraction': 0.5, 'neg_pos_ub': -1, 'add_gt_as_proposals': False}, 'allowed_border': 0, 'pos_weight': -1, 'debug': False}, 'rpn_proposal': {'nms_across_levels': False, 'nms_pre': 2000, 'nms_post': 2000, 'max_num': 2000, 'nms_thr': 0.7, 'min_bbox_size': 0}, 'rcnn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0.5, 'ignore_iof_thr': -1}, 'sampler': {'type': 'RandomSampler', 'num': 512, 'pos_fraction': 0.25, 'neg_pos_ub': -1, 'add_gt_as_proposals': True}, 'pos_weight': -1, 'debug': False}}
test_cfg = {'rpn': {'nms_across_levels': False, 'nms_pre': 1000, 'nms_post': 1000, 'max_num': 1000, 'nms_thr': 0.7, 'min_bbox_size': 0}, 'rcnn': {'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_thr': 0.5}, 'max_per_img': 100}}
neck = {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 5}
shared_head = None
pretrained = 'modelzoo://resnet50'

后面继承构造函数传入的都是这个
'''
