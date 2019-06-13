from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from .env import get_root_logger

import ipdb


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


# 该函数是主要的训练函数，用于处理单个batch数据
# data是上一层函数传入的data_batch，以字典形式包含一个batch的像素信息，图像的变换信息，gt信息等
def batch_processor(model, data, train_mode):
    losses = model(**data)      # 前向传播
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner   
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks 
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
        else:
            if cfg.data.val.type == 'CocoDataset':
                runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))
            else:
                runner.register_hook(DistEvalmAPHook(cfg.data.val))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # ipdb.set_trace()
    # prepare data loaders
    # 返回dataloader的迭代器，采用pytorch的DataLoader方法封装数据集
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus # 这里多GPU输入没用list而是迭代器，注意单GPU是range(0,1),遍历的时候只有0
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()    
    # build runner  # 这里只是将相关参数传递，没有任何的构建和运行。就像model的build一样只放了模块没有连接和顺序
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    # 注册钩子
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    # 断点加载或文件加载数据
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
