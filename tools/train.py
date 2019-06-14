from __future__ import division

import argparse
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch

import ipdb


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # 改动了：将config设置为可选择参数，这样就不用键入了，可以直接在这里改路径，方便
    parser.add_argument('--config', default='/py/mmdetection-master/configs/mask_rcnn_r101_fpn_1x.py',help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    # 断点继续训练的文件
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from') 
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    # 指定GPU数目，默认单GPU跑程序
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,      
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # 设置分布式训练，通过sh对这个py文件进行调用
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    # ipdb.set_trace()
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark  在图片输入尺度固定时开启，可以加速
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
    # if cfg.checkpoint_config is not None:
    #     # save mmdet version in checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         mmdet_version=__version__, config=cfg.text)

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

    # 模型的build和inference一样，就不多说了
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # ipdb.set_trace()
    # 注意传入的是cfg.data.train
    train_dataset = get_dataset(cfg.data.train)

    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    # 这个train只是一个入口，真正的train显然不在这里
    main()
