import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import ipdb

from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 in_channels,   # [256, 512, 1024, 2048]
                 out_channels,  # 256
                 num_outs,      # 5
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs        # 5
        self.activation = activation    # 甚至这里还能加激活函数，扩展可以说是很详尽了
        self.with_bias = normalize is None

        # 可以设置哪些层进行融合
        if end_level == -1:
            self.backbone_end_level = self.num_ins  # 4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level): #range(0,4)
            # 搭建FPN的通道变化支路，1*1卷积 将输出变为256维
            l_conv = ConvModule(    # 该模块是conv + bn/bias + activation
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            # 3*3卷积消除棋盘效应
            # 不难看出，这里虽然加了四个卷积核，但是尺寸一样（当然不能共用，因为学习的参数不同）
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        # len(inputs)描述传递过来的各stage的特征图数目
        assert len(inputs) == len(self.in_channels)
        # build laterals
        # 巧妙运用list，将input的特征图输入到lateral_convs四个通道数转化卷积层，得到256维输出
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        # 开始进行特征的自顶向下融合
        used_backbone_levels = len(laterals)
        # range(3,0,-1)步长为负递减，是为了从最高层的特征逐层向下融合
        # 将第i层最近邻插值上采样后相加融合到第i-1层，并取代其值，实现逐层融合
        # 也就是最高层的2048维输出不动，其他的三个层输出换成上采样concat的结果
        for i in range(used_backbone_levels - 1, 0, -1):    
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        # 将每个融合的层进行3*3卷积输出，没融合的2048最高层也要卷积
        # 注意：这里可以看出，是先融合再3*3消除混叠效应的，而不是边融合边卷积
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # 输出是5层但是融合只有四层，为了获得更多的特征，采用maxpooling对最高层进一步降采样，
            # 也就是tf版本实现的P6
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
