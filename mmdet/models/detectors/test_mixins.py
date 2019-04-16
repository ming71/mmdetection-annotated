from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms)

import ipdb 


class RPNTestMixin(object):
    # 注意：这里的函数处理的x其第0个维度都是img_id，也就是bs（图片的张数）
    # 一般用这个生成proposals
    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        # ipdb.set_trace()
        # 对所有特征图直接进行处理，得到RPN处理结果：
        # tuple两个元素，分别是前景背景分类和回归结果，长度按照每像素3个anchor算，每个元素的长度为特征图数目
        rpn_outs = self.rpn_head(x)     
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        # list的元素对应每张图片，元素为proposals，shape为[2000,5]，cxywh
        # 得到的proposal可以小于2000,不填充
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, img_meta, rpn_test_cfg)
            for proposals, img_meta in zip(aug_proposals, img_metas)
        ]
        return merged_proposals

class BBoxTestMixin(object):
    # 注意：这里的函数处理的x其第0个维度都是img_id，也就是bs（图片的张数）
    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        
        """Test only det bboxes without augmentation."""
        # 输入是每个bs各个图片的proposals构成的list
        # 输出是一个tensor，shape[n',5]，其中n'是所有proposal数目之和（做了整个batch的proposal的concat）
        # 前四个是xywh，第五元素舍弃排序后没用的分类得分，替代以当前proposal属于哪张图片的index(0开始)
        rois = bbox2roi(proposals)
        # 下面真正进行RoI操作：将得到的proposals RoIAlign到指定大小（7*7）
        # 输出如：[2000, 256, 7, 7]
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        # 将RoIs送入bbox_head进行分类回归得到分类分数和回归偏移,分别为[2000, 81]，[2000, 81*4]
        # 如maskrcnn的为SharedFCBBoxHead
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        # 传入rois,相当于RPN的anchor，bbox_reg是对其进行修正的
        # 精秀roi并NMS，得到真正需要的结果bbox:[N,5=cxywh]以及label:[N]
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor
                       if rescale else det_bboxes)
            # 同样将所有的bbox取出定位信息，然后原来的分类信息改为图片index，进行通道融合混合所有图片的检测目标
            mask_rois = bbox2roi([_bboxes])
            # mask rcnn在检测基础上先RoIAlign，因此使用了SingleRoIExtractor，定位到single_level文件里，再次进行RoIAlign
            # 得到的mask特征图为[100, 256, 14, 14]
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            # 卷积和转置卷积特征提取，按照类别进行打分，最后的到特征图[100, 81, 28, 28]
            mask_pred = self.mask_head(mask_feats)
            # 将得到的小特征图映射到bbox内，然后进行得分的二值化，生成一个mask，分割的地方是1,其余都是0,尺寸等同于原图大小
            # 这里就需要传入设置信息test_cfg.rcnn（test_cfg仅有两处超参数传入，一个是RPN一个是这里）
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
