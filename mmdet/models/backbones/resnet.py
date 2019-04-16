import ipdb
import logging

import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_norm_layer


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(BasicBlock, self).__init__()

        assert dcn is None, "Not implemented yet."
        #根据关键字构造不同的bn层
        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 搭建的是identity模块，其输入的plane注意，是中间的压缩的通道数（对于含downsample的略有不同）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        # ipdb.set_trace()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        self.inplanes = inplanes
        self.planes = planes
        self.normalize = normalize
        self.dcn = dcn
        self.with_dcn = dcn is not None
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        # 注意：这里打印实例对象不会显示层，因为layer赋值给了norm1，而不是self的属性，self的只赋值了name，不是层
        # norm1,2是为64通道的conv正则化，norm3是为扩展的256正则化
        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            normalize, planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(     # 这里搭建的是主通路卷积保持64通道数不变:1*1*64+3*3*64+1*1*256
            inplanes,               # 1*1*64
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1) #在这里把层加进去，打印可以显示了
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(     # 3*3*64
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(     # 1*1*256 扩展回去，便于与侧路融合
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        # print(inplanes,planes)
        self.relu = nn.ReLU(inplace=True)   #再加ReLU
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.normalize = normalize

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            # ipdb.set_trace()
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

# 64 64
# 128 256
# 256 512
# 512 1024

def make_res_layer(block,   #Bottleneck类
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   normalize=dict(type='BN'),
                   dcn=None):
    downsample = None
    # ipdb.set_trace()
    '''
    下面的判断语句工作方式(resnet-101)：
    1.对于stride=1,2,2,2,在第一层layer第一个block上用到or后的条件，即输入通道64,输出通道也是64，不为4倍关系，添加dowmsample
    2.修改inplanes为64*4=256,创建后面的若干identity block，因为后面的输入输出都是一样的
    2.第一个layer的后面block，直接append创建bottleneck
    3.对于layer1之后的stage，stride=2,第一个block创建downsample
    4.layer_i的后面的block，直接append添加block
    '''
    if stride != 1 or inplanes != planes * block.expansion: #判断是否在skip connect上加卷积层bn
        # ipdb.set_trace()
        downsample = nn.Sequential(
            nn.Conv2d(      #256个1*1卷积压缩
                inplanes,
                planes * block.expansion, #
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(normalize, planes * block.expansion)[1],   #返回的0是name，1是layer
        )

    layers = []
    layers.append(  #创建分支归分支，该路径的Bottleneck也是要搭建的，不过由于其输入输出的通道比较不同，单独加入
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp,
            normalize=normalize,
            dcn=dcn))
    # 修改下一级的输入很关键，下面就是纯粹的identity模块
    inplanes = planes * block.expansion 

    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                style=style,
                with_cp=with_cp,
                normalize=normalize,
                dcn=dcn))

    return nn.Sequential(*layers)

#分析以ResNet-101为例
@BACKBONES.register_module
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    # 只提供五种不同深度的backbone： ResNet=18/34/50/101/152
    # 通过四种depth索引，value是tuple
    # 后面的元组设置的是各个block的个数
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 normalize=dict(type='BN', frozen=False),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.normalize = normalize
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        # ipdb.set_trace()

        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.zero_init_residual = zero_init_residual
        # block为BasicBlock/Bottleneck
        self.block, stage_blocks = self.arch_settings[depth]
        # 通过key深度索引五种不同的resnet框架 self.stage_blocks：(3, 4, 23, 3)
        self.stage_blocks = stage_blocks[:num_stages]   #通过key深度索引五种不同的resnet框架
        self.inplanes = 64

        self._make_stem_layer() #搭建模型开始的部分

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            # ipdb.set_trace()
            # num_blocks遍历(3, 4, 23, 3),每次按照这个数目进行block模块堆叠
            # 逐次遍历strides=(1, 2, 2, 2)  dilations = (1, 1, 1, 1)
            stride = strides[i]     
            dilation = dilations[i] 
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = 64 * 2**i   # 64,128,256,512  
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                normalize=normalize,
                dcn=dcn)
            # print(planes,self.inplanes)
            self.inplanes = planes * self.block.expansion   #64
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        # ipdb.set_trace()

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1) #最终输出的维度，2048

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        # 注意torch的动态图是在计算时构建的，这里只是将每个层作为self的属性添加到类实例self中
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1_name, norm1 = build_norm_layer(  #搭建bn层，可以设定gn等，暂时不看
            self.normalize, 64, postfix=1)
        self.add_module(self.norm1_name, norm1) 
        self.relu = nn.ReLU(inplace=True)   #直接将输出覆盖输入节省内存
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        #只要设置了冻结，那么先阻断layer之前最开始的层的梯度
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False  #阻断梯度的更新
        #冻结flag及其之前的层不更新参数
        for i in range(1, self.frozen_stages + 1):
            # ipdb.set_trace()
            m = getattr(self, 'layer{}'.format(i))  
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x) #在stage1之前的预处理层
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)   #取出每个stage的layer
            x = res_layer(x)
            if i in self.out_indices:   #把每个阶段的输出，即前向计算中间特征进行保存，用于FPN融合
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
