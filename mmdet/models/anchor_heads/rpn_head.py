import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from .anchor_head import AnchorHead
from ..registry import HEADS

import ipdb


# 搭建模型后，针对不同尺度（5个）会执行5次forward
@HEADS.register_module
class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        # 先传个2进去，因为rpn用来前景背景二分类，再丢个输入通道剩下的直接都扔进去初始化
        super(RPNHead, self).__init__(2, in_channels, **kwargs)
        # 在这里init后已经添加完rpn model了 

    # 在AnchorHead中调用到，用于搭建网络，简单的三层网络
    def _init_layers(self):
        # 通道256-->256提取特征
        self.rpn_conv = nn.Conv2d(      
            self.in_channels, self.feat_channels, 3, padding=1)
        # sigmoid前景背景二分类，self.cls_out_channels=1,定义在AnchorHead里
        # 输出通道为3，每个anchor一个参数
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        # 并行进行回归，输出通道12,每个anchor四个参数
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)


    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        # ipdb.set_trace()
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_reg=losses['loss_reg'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        # 遍历每一张特征图得到的anchors进行坐标还原
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # 按照得分从上万个里面选取2000个最高得分的box
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # 偏移转换为真实的bbox作标
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            # NMS筛选
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)    #将得加入xywh中成为xywhc，便于NMS
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)    #每张特征图选完得到的box都加到这里
        # 全部的proposal box连接到一起
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            # proposal再按照分类得分压缩到2000（如果本身就小于2000就只排序）
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])  # proposal的数目
            _, topk_inds = scores.topk(num)             # 按照score排序
            proposals = proposals[topk_inds, :]
        return proposals
