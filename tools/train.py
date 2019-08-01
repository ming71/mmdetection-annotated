from __future__ import division

import argparse
import os
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import build_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch

import ipdb


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # 改动了：将config设置为可选择参数，这样就不用键入了，可以直接在这里改路径，方便
    parser.add_argument('--config', help='train config file path',default='../configs/faster_rcnn_r50_fpn_1x.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark 
    # 在图片输入尺度固定时开启，可以加速.一般都是关的，只有在固定尺度的网络如SSD512中才开启
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        # 创建工作目录存放训练文件，如果不键入，会自动按照py配置文件生成对应的目录
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:    
        # 断点继续训练的权值文件
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # ipdb.set_trace(context=35)
    #  搭建模型
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

     # 将训练配置传入
    train_dataset = build_dataset(cfg.data.train)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in checkpoints as meta data
        # 要注意的是，以前发布的模型是不存这个类别等信息的，
        # 用的默认COCO或者VOC参数，所以如果用以前训练好的模型检测时会提醒warning一下，无伤大雅
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)

    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES   # model的CLASSES属性本来没有的，但是python不用提前声明，再赋值的时候自动定义变量
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()

