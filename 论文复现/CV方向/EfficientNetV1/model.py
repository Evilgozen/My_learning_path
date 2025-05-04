import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

#常用于MoblieNet等中对于硬件的二进制下保证通道个数的优化
def _make_divisible(ch,divisor=8,min_ch=None):
    #默认大小
    if min_ch is None:
        min_ch=divisor
    new_ch=max(min_ch,int(ch+divisor / 2) // divisor * divisor) #对于上取整
    #防止太小
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

#配置drop_out的参数实现,
def drop_path(x,drop_prob:float=0.,training: bool = False):
    if drop_prob == 0. or not training: #边界处理
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) #这个shape0实际上是batch，然后后面是需要做缩放的值
    random_tensor = keep_prob + torch.rand(shape,dtype=x.dtype,device=x.device) #[kepp_prob,1+keep_prob]的值
    random_tensor.floor() #大于一的等于一小于一的变成零
    output = x.div(keep_prob) * random_tensor #除以deep_prob放缩再mask掉
    return output #最后类似于丢弃输入中的部分batch计算路径的效果

#继承Module，将drop_out实现为层结构
class DropPath(nn.Module):
    def __init__(self,drop_prob=None):
        super(DropPath,self).__init__()
        self.drop_prob=drop_prob

    def forward(self,x):
        return drop_path(x,self.drop_prob,self.training)

#BN层，在是线上还存在升降维的作用
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes:int,     #输入通道的大小
                 out_planes:int,    #输出通道大小
                 kernel_size:int=3, #卷积核大小(默认3*3)
                 stride:int=1,      #步长(默认1）
                 groups:int=1,      #分组卷积参数(默认1，普通卷积),如果是in_planes则为dw卷积
                 norm_layer:Optional[Callable[...,nn.Module]] = None, #归一化层
                 activation_layer:Optional[Callable[...,nn.Module]] = None): #激活函数
        padding = (kernel_size - 1) // 2 #same卷积自动计算padding的大小
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU #也叫做Swish激活函数

        super(ConvBNActivation,self).__init__(  #调用父类构造函数继承
            nn.Conv2d(in_channels=in_planes,
                      out_channels=out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False), #卷积层(bias=False,因为后面是BN
            norm_layer(out_planes),
            activation_layer()
        )

class SqueezeExcitation(nn.Module): #se注意力块
    def __init__(self,
                 input_c:int,
                 expand_c:int,
                 squeeze_factor:int = 4):
        super(SqueezeExcitation,self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c,squeeze_c,1) #1x1做的降维,将通道降低为1/4
        self.ac1 = nn.SiLU() #激活函数swish
        self.fc2 = nn.Conv2d(squeeze_c,expand_c,1) #升维
        self.ac2 = nn.Sigmoid()

    def forward(self,x:Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x,output_size=(1,1)) #平均池化，1*1池化？不就是求了个均值吗？ 答案是对，但是分了chanel然后放缩好用
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x #最后相当于对于chanel的权重重新分配，也就是为什么要一个sigmoid

#核心模块-倒残差块
class InvertedResidualConfig: #basline实际上是MobileNetV3
    def __init__(self,
                 kernel:int, #dw卷积的kernel
                 input_c:int,
                 out_c:int,
                 expanded_ratio:int, #对于倒残差的放缩(因为dw卷积，导致chanel由^2变为正相关)
                 stride:int, #dw卷积可能为1或者2的步距
                 use_se:int, #用不用注意力，但是实际上一直True
                 drop_rate:float, #drop_path
                 index:str, #日志debug 1a,2a,2b
                width_coefficient:float): #EfficientNet的宽度超参
        self.input_c = self.adjust_channels(input_c, width_coefficient) #初始化为B0，后续通过调width_coefficient更改
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio #就是倒残差的升维
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels:int,width_cofficient:float):
        return _make_divisible(channels * width_cofficient,8)

#完整的一个MBNet的块结构，包括倒残差的BN升维+dw卷积(升维减去了PW升维的层)+BN降维(保证捷径分支的作用)[可选是否做一个SE注意力层]
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf:InvertedResidualConfig,
                 norm_layer:Callable[...,nn.Module]):
        super(InvertedResidual,self).__init__()

        if cnf.stride not in [1,2]:
            raise ValueError("stride 违法")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c==cnf.out_c) #符不符合残差原理

        layers = OrderedDict() #有序字典搭建层结构
        activation_layer = nn.SiLU

        #具体的层的堆叠
        if cnf.expanded_c != cnf.input_c: #如果需要升维
            layers.update({"expand_Conv":ConvBNActivation(cnf.input_c,  #BN做个升维度
                                                          cnf.expanded_c,
                                                          kernel_size=1,
                                                          norm_layer=norm_layer,
                                                          activation_layer=activation_layer)})
        #dw卷积
        layers.update({"dwConv":ConvBNActivation(cnf.expanded_c, #输入的channels
                                                 cnf.expanded_c,
                                                 kernel_size=cnf.kernel,
                                                 groups=cnf.expanded_c, #nb
                                                 norm_layer=norm_layer,
                                                 activation_layer=activation_layer)})

        if cnf.use_se: #做一个注意力分配
            layers.update({"se":SqueezeExcitation(cnf.input_c,
                                                  cnf.expanded_c)})

        layers.update({"project_Conv":ConvBNActivation(cnf.expanded_c,
                                                       cnf.out_c,
                                                       kernel_size=cnf.kernel,
                                                       norm_layer=norm_layer,
                                                       activation_layer=activation_layer)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        #dropPath层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity() #占位

    def forward(self,x:Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result

class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient:float,
                 depth_coefficient:float,
                 num_classes:int = 1000,
                 dropout_rate:float = 0.2, #最后GAP下采样FC的dropout
                 drop_connect_rate:float = 0.2, #drop_path的比例
                 block:Optional[Callable[...,nn.Module]]=None,
                 norm_layer:Optional[Callable[...,nn.Module]]=None):
        super(EfficientNet,self).__init__()

        #参数：核大小(dw卷积的核),输入通道，输出通道，？，步距，注意力层，drop率，重复次数
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d,eps=1e-3,momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=)

        b=0
        num_blocks