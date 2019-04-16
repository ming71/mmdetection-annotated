from .two_stage import TwoStageDetector
from ..registry import DETECTORS
import ipdb

# 利用super的构造函数继承，递归地搭建了计算图！
# 依次递归到父类去build初始化
# 递归继承中，只有Module类继承构造函数，其他的继承的是定制函数
@DETECTORS.register_module
class MaskRCNN(TwoStageDetector):
    # ipdb.set_trace()
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        # ipdb.set_trace()
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        # ipdb.set_trace()

#传入这里的参数是：
'''
self = MaskRCNN()
backbone = {'type': 'ResNet', 'depth': 101, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'style': 'pytorch'}
neck = {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 5}
rpn_head = {'type': 'RPNHead', 'in_channels': 256, 'feat_channels': 256, 'anchor_scales': [8], 'anchor_ratios': [0.5, 1.0, 2.0], 'anchor_strides': [4, 8, 16, 32, 64], 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [1.0, 1.0, 1.0, 1.0], 'use_sigmoid_cls': True}
bbox_roi_extractor = {'type': 'SingleRoIExtractor', 'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 2}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}
bbox_head = {'type': 'SharedFCBBoxHead', 'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 81, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.1, 0.1, 0.2, 0.2], 'reg_class_agnostic': False}
mask_roi_extractor = {'type': 'SingleRoIExtractor', 'roi_layer': {'type': 'RoIAlign', 'out_size': 14, 'sample_num': 2}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}
mask_head = {'type': 'FCNMaskHead', 'num_convs': 4, 'in_channels': 256, 'conv_out_channels': 256, 'num_classes': 81}
train_cfg = None
test_cfg = {'rpn': {'nms_across_levels': False, 'nms_pre': 2000, 'nms_post': 2000, 'max_num': 2000, 'nms_thr': 0.7, 'min_bbox_size': 0}, 'rcnn': {'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_thr': 0.5}, 'max_per_img': 100, 'mask_thr_binary': 0.5}}
pretrained = None
后面继承构造函数传入的都是这个
'''


