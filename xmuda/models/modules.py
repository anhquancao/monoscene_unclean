import torch
import torch.nn as nn
from xmuda.models.DDR import Bottleneck3D
import numpy as np
from xmuda.models.resnet34_unet import UNetResNet34


class ProcessKitti(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[Bottleneck3D(feature, 
                           feature // 4, 
                           bn_momentum=bn_momentum, 
                           norm_layer=norm_layer, 
                           dilation=[i, i, i]) for i in dilations]
        )
    def forward(self, x):
        return self.main(x)

class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[Bottleneck3D(feature, 
                           feature // 4, 
                           bn_momentum=bn_momentum, 
                           norm_layer=norm_layer, 
                           dilation=[i, i, i]) for i in dilations]
        )
    def forward(self, x):
        return self.main(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential( 
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1), 
            norm_layer(out_channels, momentum=bn_momentum), 
            nn.ReLU()
        )

    def forward(self, x):
        return self.main(x)

class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(feature, 
                                 feature // 4, 
                                 bn_momentum=bn_momentum, 
                                 expansion=expansion, stride=2, 
                                 downsample=nn.Sequential(
                                     nn.AvgPool3d(kernel_size=2, stride=2), 
                                     nn.Conv3d(feature, int(feature * expansion/4), kernel_size=1, stride=1, bias=False), 
                                     norm_layer(int(feature * expansion/4), momentum=bn_momentum)
                                 ),
                                 norm_layer=norm_layer)

    def forward(self, x):
        return self.main(x)

