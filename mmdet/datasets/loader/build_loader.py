from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader

from .sampler import GroupSampler, DistributedGroupSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedGroupSampler(dataset, imgs_per_gpu, world_size,
                                          rank)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else: 
        # 正常训练看这里
        if not kwargs.get('shuffle', True):
            sampler = None
        else:
            sampler = GroupSampler(dataset, imgs_per_gpu)   # batch中样本的采样方式
        batch_size = num_gpus * imgs_per_gpu    # 在这里定义batch size
        num_workers = num_gpus * workers_per_gpu # 多线程读取可以加快数据的读取速度

    # 采用pytorch内置的DataLoader方法
    # DataLoader是一个 迭代器
    # dataset继承自最高父类pytorch的Dataset；num_workers：使用多进程加载的进程数，0代表不使用多进程 ；pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些 ；
    # collate_fn：在数据处理中，有时会出现某个样本无法读取等问题，比如某张图片损坏。
    #             这时在_ getitem _函数中将出现异常，此时最好的解决方案即是将出错的样本剔除。
    #             如果实在是遇到这种情况无法处理，则可以返回None对象，然后在Dataloader中实现自定义的collate_fn，将空对象过滤掉。
    #             但要注意，在这种情况下dataloader返回的batch数目会少于batch_size。

    # sampler：自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader
