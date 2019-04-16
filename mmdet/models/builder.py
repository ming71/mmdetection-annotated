import mmcv
from torch import nn
import ipdb

#此处不会在执行registry而是直接进行sys.modules查询得到
from .registry import BACKBONES, NECKS, ROI_EXTRACTORS, HEADS, DETECTORS   

#这里不仅build detectors，还会完成其相关组件的搭建
def _build_module(cfg, registry, default_args):
    # ipdb.set_trace()
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()   #浅拷贝
    #args剔除typename剩下pretrained+model_settings,返回模型名obj_type如Mask RCNN
    obj_type = args.pop('type') 
    if mmcv.is_str(obj_type):
        if obj_type not in registry.module_dict:    
            #注册的DETECTOR中必须包含此type，否则自定义编写并加入
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
            # 这里的registry的_module_dict属性中包含的是detector下的模型type
            # 索引key得到相应的class
            # 注意了：obj_type在这里不再只是一个字符而是一个class
        obj_type = registry.module_dict[obj_type]   #通过type的字符索引字典同名的value得到对应的class
    elif not isinstance(obj_type, type):    #输入不能为空
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():    #items()返回字典的键值对用于遍历
            args.setdefault(name, value)    #将default_args的键值对加入到args中，将模型和训练配置进行整合送入类中
    # 注意：无论训练/检测，都会build DETECTORS，
    # 而该Registry中的_module_dict字典已经提前在import阶段加入了所有其下对应模型的class,
    # 这里传入obj_type（如maskrcnn）class的参数为其进行实例化
    # 也就是说，这里是进入各个需要build，类实例化的入口，从这里链接到各个模块的类，如MaskRCNN,ResNet,FPN等
    # ipdb.set_trace()
    return obj_type(**args)
    #**args是将字典unpack得到各个元素，分别与形参匹配送入函数中


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
