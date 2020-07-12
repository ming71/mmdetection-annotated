[TOC]

### 代码注释
#### config files

##### anchor setting

head中的anchor设置参数为：

```
octave_base_scale=4,
scales_per_octave=3,
anchor_ratios=[0.5, 1.0, 2.0],
anchor_strides=[8, 16, 32, 64, 128],
```

其中`anchor_strides`代表的是anchor感受野，降采样步长，`scales_per_octave`控制尺度，`octave_base_scale`是anchor面积的开方sqrt(wh)，和ratio一起生成anchor。每个位置的anchor数目是scales_per_octave*anchor_ratios。具体而言，例如stride为32时，生成九个anchor为：

```
(32*8*（2^(0/3)）/sqrt(2), 32*8（2^(0/3)）* sqrt(2)), 
(32*8*（2^(1/3)）/sqrt(2), 32*8（2^(1/3)）* sqrt(2)),
(32*8*（2^(2/3)）/sqrt(2), 32*8（2^(2/3)）* sqrt(2));
(32*8*（2^(0/3)）/sqrt(1), 32*8（2^(0/3)）* sqrt(1)),
(32*8*（2^(1/3)）/sqrt(1), 32*8（2^(1/3)）* sqrt(1)),
(32*8*（2^(2/3)）/sqrt(1), 32*8（2^(2/3)）* sqrt(1));
(32*8*（2^(0/3)）/sqrt(0.5), 32*8（2^(0/3)）* sqrt(0.5)), 
(32*8*（2^(1/3)）/sqrt(0.5), 32*8（2^(1/3)）* sqrt(0.5)),
(32*8*（2^(2/3)）/sqrt(0.5), 32*8（2^(2/3)）* sqrt(0.5)).
```

很明白了吧，scales_per_octave的个数代表`2^0`开始的多少个尺度，等分1作为幂指数；anchor_ratios和scales_per_octave其实推算一下不难得出：$h= s\sqrt{r} , w= s/ \sqrt{r} $，其中s就是octave_base_scale，r就是ratio了，wh顺序无所谓，反正对称的。

---

### 问题合集

#### 历史版本问题

* RuntimeError: any only supports torch.uint8 dtype

```
RuntimeError: any only supports torch.uint8 dtype
```

数据类型问题，可以参考[这里](https://github.com/open-mmlab/mmdetection/pull/2939/commits/ccc9fe69a37ddfbce48f37ac9c4dc56346d6589d)，将`mmdet/core/anchor/utils.py`作如下改动：

```
return inside_flags
改为
return inside_flags.to(torch.bool)
```

* assert len(indices) == self.num_samples

```
assert len(indices) == self.num_samples
```

出现原因是图片数除以img_per_GPU不得整数，参考[这里](https://github.com/open-mmlab/mmdetection/pull/1610/commits/1e1297897b46f094db13e15849a8cfd0942bcb6d)，将`mmdet/datasets/loader/sampler.py`的如下代码段进行修改即可：

```
indice = np.concatenate([indice, indice[:num_extra]])
改为；
indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
```



#### 其他问题