# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict
from xmuda.models.context_prior import ContextPrior3D
#from config import config
#from resnet import get_resnet50
#

class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out


'''
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
'''
class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu


class Decoder3D(nn.Module):
    def __init__(self, class_num, norm_layer, in_channels=[256, 128, 128], feature=128, 
                 bn_momentum=0.1):
        super(Decoder3D, self).__init__()
        self.business_layer = []

        self.in_channels = in_channels
        self.feature = feature

        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.business_layer.append(self.pooling)

        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1)

        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2)

        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            )]
        )
        self.seg_head_1_8 = nn.Sequential(
            nn.Dropout3d(.1),
            nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
        )

        self.seg_head_1_16 = nn.Sequential(
            nn.Dropout3d(.1),
            nn.Conv3d(feature * 2, class_num, kernel_size=1, bias=True)
        )

        self.business_layer.append(self.classify_semantic)

        self.enc_1_8 = nn.Sequential(
#            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(in_channels['1_8'], in_channels['1_8'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(in_channels['1_8'], in_channels['1_8'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(in_channels['1_8'], in_channels['1_8'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.enc_1_16 = nn.Sequential(
#            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(in_channels['1_16'], in_channels['1_16'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(in_channels['1_16'], in_channels['1_16'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(in_channels['1_16'], in_channels['1_16'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.resize_1_8 = nn.Conv3d(feature + in_channels['1_8'], feature, kernel_size=1)
        self.resize_1_16 = nn.Conv3d(feature * 2 + in_channels['1_16'], feature * 2, kernel_size=1)
            

    def forward(self, input_dict):
        x3d_input_1_4 = input_dict['x3d_1_4']
        x3d_input_1_8 = input_dict['x3d_1_8']
        x3d_input_1_16 = input_dict['x3d_1_16']
#        if self.in_channels != self.feature:
#            x3d = self.downsample(x3d)

        x3d_1_8 = self.semantic_layer1(x3d_input_1_4)
        x3d_input_1_8 = self.enc_1_8(x3d_input_1_8)
        x3d_1_8 = torch.cat([x3d_1_8, x3d_input_1_8], dim=1)
        x3d_1_8 = self.resize_1_8(x3d_1_8)

        x3d_1_16 = self.semantic_layer2(x3d_1_8)
        x3d_input_1_16 = self.enc_1_16(x3d_input_1_16)
        x3d_1_16 = torch.cat([x3d_1_16, x3d_input_1_16], dim=1)
        x3d_1_16 = self.resize_1_16(x3d_1_16)
        seg_logit_1_16 = self.seg_head_1_16(x3d_1_16)
#        print("1_16", x3d_1_16.shape)

        x3d_up_1_8 = self.classify_semantic[0](x3d_1_16)
        x3d_up_1_8 = x3d_up_1_8 + x3d_1_8
        seg_logit_1_8 = self.seg_head_1_8(x3d_up_1_8)
#        print("1_8", x3d_up_1_8.shape)

        x3d_up_1_4 = self.classify_semantic[1](x3d_up_1_8)
        seg_logit_1_4 = self.classify_semantic[2](x3d_up_1_4)

        return {
            "ssc_logit_1_4": seg_logit_1_4,
            "ssc_logit_1_8": seg_logit_1_8,
            "ssc_logit_1_16": seg_logit_1_16
        }


'''
main network
'''
class Network(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(Network, self).__init__()
        self.business_layer = []

        if eval:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=config.bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=nn.BatchNorm2d)
        else:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=config.bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=norm_layer)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.stage1 = STAGE1(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage1.business_layer

        self.stage2 = STAGE2(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage2.business_layer

    def forward(self, rgb, depth_mapping_3d, tsdf, sketch_gt=None):

        h, w = rgb.size(2), rgb.size(3)

        feature2d = self.backbone(rgb)

        pred_sketch_raw, pred_sketch_gsnn, pred_sketch, pred_mean, pred_log_var = self.stage1(tsdf, depth_mapping_3d, sketch_gt)
        pred_semantic, _ = self.stage2(feature2d, depth_mapping_3d, pred_sketch_raw,
                                                                pred_sketch_gsnn)

        if self.training:
            return pred_semantic, _, pred_sketch_raw, pred_sketch_gsnn, pred_sketch, pred_mean, pred_log_var
        return pred_semantic, _, pred_sketch_gsnn

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == '__main__':
    model = Network(class_num=12, norm_layer=nn.BatchNorm3d, feature=128, eval=True)
    # print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    left = torch.rand(1, 3, 480, 640).cuda()
    right = torch.rand(1, 3, 480, 640).cuda()
    depth_mapping_3d = torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
    tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

    out = model(left, depth_mapping_3d, tsdf, None)
