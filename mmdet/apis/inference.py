import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import cv2

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform

import ipdb


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    # 执行img_transform实例的call方法，直接调用
    # 进行图片的归一化/翻转等增强
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    # 写入img_meta相关信息
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])

# 预测单张图片
def _inference_single(model, img, img_transform, cfg, device):
    # ipdb.set_trace()
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():
        # **data是字典传入包含img和img_meta的信息
        # 此处一定要传入是否训练的标志return_loss=False
        result = model(return_loss=False, rescale=True, **data)
    return result

# 多张图片迭代单张图片返回一个生成器
def _inference_generator(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)


def inference_detector(model, imgs, cfg, device='cuda:0')   :
    img_transform = ImageTransform(     # 这个类实现了__call__放法，img_transform 可调用
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):      # 如果是list说明是多张图片
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:                           # 否则单张图
        return _inference_generator(model, imgs, img_transform, cfg, device)


def show_result(img, result, dataset='coco', score_thr=0.3, out_file=None):
    # ipdb.set_trace()
    img = mmcv.imread(img)
    class_names = get_classes(dataset)  # 获取类名list
    # 返回的tuple均含detection，看情况是否包含mask
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    # draw segmentation masks
    # 这里的img已经加上了mask，可以直接输出了
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # 直接显示mask结果
    # cv2.imshow('window',img)
    # cv2.waitKey(0)

    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # 显示box
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None)
