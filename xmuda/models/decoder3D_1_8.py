# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xmuda.models.LMSCNet import SegmentationHead
from xmuda.models.context_prior import ContextPrior3D
from xmuda.models.context_prior_v2 import ContextPrior3Dv2
from xmuda.models.CP_baseline import CPBaseline
from xmuda.models.CP_implicit import CPImplicit
from xmuda.models.CP_implicit_pairwise import CPImplicitPairwise
#from xmuda.models.CP_implicit_pairwise_random import CPImplicitPairwise
#from xmuda.models.CP_implicit_4SetsOfk import CPImplicitPairwise
#from xmuda.models.CP_implicit_pairwise_v2 import CPImplicitPairwise
from xmuda.models.CP_implicit_leftnonempty import CPImplicitV2
from xmuda.models.DDR import Bottleneck3D
from functools import partial
from collections import OrderedDict


class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
    def forward(self, x):
        return self.main(x)

class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(feature, 
                                 feature // 4, 
                                 bn_momentum=bn_momentum, 
                                 expansion=8, stride=2, 
                                 downsample=nn.Sequential(
                                     nn.AvgPool3d(kernel_size=2, stride=2), 
                                     nn.Conv3d(feature, feature * 2, kernel_size=1, stride=1, bias=False), 
                                     norm_layer(feature * 2, momentum=bn_momentum)
                                 ), 
                                 norm_layer=norm_layer)

    def forward(self, x):
        return self.main(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential( 
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1), 
            norm_layer(out_channels, momentum=bn_momentum), 
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.main(x)


class ResizeInterpolate(nn.Module):
    def __init__(self, in_channels, out_channels, target_size, norm_layer, bn_momentum):
        super(ResizeInterpolate, self).__init__()
        self.target_size = target_size
        self._net = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  norm_layer(out_channels, momentum=bn_momentum),
                                  nn.LeakyReLU())

    def forward(self, x):
        f = F.interpolate(x, size=self.target_size, mode='trilinear', align_corners=True)
        return self._net(f)


class Merge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Merge, self).__init__()
        self.resize = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, xs):
        return self.resize(torch.cat(xs, dim=1))


class Decoder3D(nn.Module):
    def __init__(self, class_num, norm_layer,
                 max_k=128,
                 project_res_6_12=True,
                 project_res_8_16=True,
                 context_prior=None,
                 in_channels={'1_16': 256, '1_8': 128, '1_4': 128},
                 feature=128,
                 skip_level=1,
                 bn_momentum=0.1):
        super(Decoder3D, self).__init__()
        self.business_layer = []
        
        self.skip_level = skip_level

        self.in_channels = in_channels
        self.feature = feature
        self.feature_1_4 = 64
        self.feature_1_6 = 96
        self.feature_1_8 = 128
        self.feature_1_12 = 192
        self.feature_1_16 = 256

        self.feature_1_16_dec = 768
        self.feature_1_12_dec = 384
        self.feature_1_8_dec = 256
        self.feature_1_6_dec = 192
        self.feature_1_4_dec = 128
        self.size_1_4 = (60, 36, 60)
        self.size_1_6 = (40, 24, 40)
        self.size_1_8 = (30, 18, 30)
        self.size_1_12 = (20, 12, 20)
        self.size_1_16 = (15, 9, 15)

        self.project_res_6_12 = project_res_6_12
        self.project_res_8_16 = project_res_8_16

#        self.resize_input_1_4 = nn.Conv3d(self.feature_1_4 + 3, self.feature_1_4, kernel_size=1)
#        self.resize_input_1_6 = nn.Conv3d(self.feature_1_6 + 3, self.feature_1_6, kernel_size=1)
#        self.resize_input_1_8 = nn.Conv3d(self.feature_1_8 + 3, self.feature_1_8, kernel_size=1)
#        self.resize_input_1_12 = nn.Conv3d(self.feature_1_12 + 3, self.feature_1_12, kernel_size=1)
#        self.resize_input_1_16 = nn.Conv3d(self.feature_1_16 + 3, self.feature_1_16, kernel_size=1)

        self.process_1_4 = Process(self.feature_1_4, norm_layer, bn_momentum)
        self.process_1_6 = Process(self.feature_1_6, norm_layer, bn_momentum)
        self.process_1_8 = Process(self.feature_1_8, norm_layer, bn_momentum)
        self.process_1_12 = Process(self.feature_1_12, norm_layer, bn_momentum)
        self.process_1_16 = Process(self.feature_1_16, norm_layer, bn_momentum)


        self.down_4_6 = ResizeInterpolate(self.feature_1_4, self.feature_1_6, self.size_1_6, norm_layer, bn_momentum)

        self.down_6_8 = ResizeInterpolate(self.feature_1_6, self.feature_1_8, self.size_1_8, norm_layer, bn_momentum)

        self.down_8_12 = ResizeInterpolate(self.feature_1_8, self.feature_1_12, self.size_1_12, norm_layer, bn_momentum)

        if self.skip_level >= 2:
            self.down_4_6_8 = ResizeInterpolate(self.feature_1_6, self.feature_1_8, self.size_1_8, norm_layer, bn_momentum)
            self.down_6_8_12 = ResizeInterpolate(self.feature_1_8, self.feature_1_12, self.size_1_12, norm_layer, bn_momentum)
            self.down_8_12_16 = ResizeInterpolate(self.feature_1_12, self.feature_1_16, self.size_1_16, norm_layer, bn_momentum)
        if self.skip_level >= 3:
            self.down_4_6_8_12 = ResizeInterpolate(self.feature_1_8, self.feature_1_12, self.size_1_12, norm_layer, bn_momentum)
            self.down_6_8_12_16 = ResizeInterpolate(self.feature_1_12, self.feature_1_16, self.size_1_16, norm_layer, bn_momentum)
        if self.skip_level >= 4:
            self.down_4_6_8_12_16 = ResizeInterpolate(self.feature_1_12, self.feature_1_16, self.size_1_16, norm_layer, bn_momentum)

        self.down_12_16 = ResizeInterpolate(self.feature_1_12, self.feature_1_16, self.size_1_16, norm_layer, bn_momentum)


        self.process_1_16_dec = Process(self.feature_1_16_dec, norm_layer, bn_momentum)

        self.up_1_16_1_12 = ResizeInterpolate(self.feature_1_16_dec, self.feature_1_12_dec, self.size_1_12, norm_layer, bn_momentum)
        self.up_1_12_1_8 = ResizeInterpolate(self.feature_1_12_dec, self.feature_1_8_dec, self.size_1_8, norm_layer, bn_momentum)
        self.up_1_8_1_6 = ResizeInterpolate(self.feature_1_8_dec, self.feature_1_6_dec, self.size_1_6, norm_layer, bn_momentum)
        self.up_1_6_1_4 = ResizeInterpolate(self.feature_1_6_dec, self.feature_1_4_dec, self.size_1_4, norm_layer, bn_momentum)

        merge_1_6_input_size = self.feature_1_6
        merge_1_12_input_size = self.feature_1_12
        merge_1_8_input_size = self.feature_1_8
        merge_1_16_input_size = self.feature_1_16

        if self.skip_level >= 2:
            merge_1_8_input_size += self.feature_1_8
            merge_1_12_input_size += self.feature_1_12
            merge_1_16_input_size += self.feature_1_16
        if self.skip_level >= 3:
            merge_1_12_input_size += self.feature_1_12
            merge_1_16_input_size += self.feature_1_16
        if self.skip_level >= 4:
            merge_1_16_input_size += self.feature_1_16

        if project_res_6_12:
            merge_1_6_input_size += self.feature_1_6
            merge_1_12_input_size += self.feature_1_12
        self.merge_1_6_enc = Merge(merge_1_6_input_size, self.feature_1_6)
        self.merge_1_12_enc = Merge(merge_1_12_input_size, self.feature_1_12)

        if project_res_8_16:
            merge_1_8_input_size += self.feature_1_8
            merge_1_16_input_size += self.feature_1_16

        self.merge_1_8_enc = Merge(merge_1_8_input_size, self.feature_1_8)
        self.merge_1_16_enc = Merge(merge_1_16_input_size, self.feature_1_16)

        self.merge_1_4_dec = Merge(self.feature_1_4_dec + self.feature_1_4, self.feature_1_4_dec)
        self.merge_1_6_dec = Merge(self.feature_1_6_dec + self.feature_1_6, self.feature_1_6_dec)
        self.merge_1_8_dec = Merge(self.feature_1_8_dec + self.feature_1_8, self.feature_1_8_dec)
        self.merge_1_12_dec = Merge(self.feature_1_12_dec + self.feature_1_12, self.feature_1_12_dec)
#        self.merge_1_16_dec = Merge(self.feature_1_16 + self.feature_1_16_dec, self.feature_1_16_dec)

        self.ssc_head_1_4 = nn.Sequential( 
            nn.Dropout3d(.1), 
            SegmentationHead(self.feature_1_4_dec, self.feature_1_4_dec, class_num, [1, 2, 3])
        )
        self.ASPP_1_16 = SegmentationHead(self.feature_1_16, self.feature_1_16, self.feature_1_16, [1, 2, 3])
#        self.ssc_head_1_16 = nn.Sequential( 
#            nn.Dropout3d(.1), 
#            SegmentationHead(self.feature_1_16, self.feature_1_16, class_num, [1, 2, 3])
#        )
        self.context_prior = context_prior
        if context_prior == "CRCP":
            self.CP_implicit_pairwise = CPImplicitPairwise(self.feature_1_8, self.feature_1_8_dec, 
                                                           self.feature_1_8, (30, 18, 30), 
                                                           max_k=max_k, 
                                                           n_classes=class_num, 
                                                           bn_momentum=bn_momentum)

    def forward(self, input_dict):
        x3d_input_1_4 = input_dict['x3d_1_4']
        x3d_input_1_6 = input_dict['x3d_1_6']
        x3d_input_1_8 = input_dict['x3d_1_8']
        x3d_input_1_12 = input_dict['x3d_1_12']
        x3d_input_1_16 = input_dict['x3d_1_16']
        pts_cam_1_16 = input_dict['pts_cam_1_16']
        pts_cam_1_8 = input_dict['pts_cam_1_8']
        max_k = input_dict['max_k']
        res = {}

        x3d_1_4 = self.process_1_4(x3d_input_1_4)
        x3d_1_4_down_1_6 = self.down_4_6(x3d_1_4)
        if self.skip_level >= 2:
            x3d_1_4_down_1_8 = self.down_4_6_8(x3d_1_4_down_1_6)
        if self.skip_level >= 3:
            x3d_1_4_down_1_12 = self.down_4_6_8_12(x3d_1_4_down_1_8)
        if self.skip_level >= 4:
           x3d_1_4_down_1_16 = self.down_4_6_8_12_16(x3d_1_4_down_1_12)

        input_1_6s = [x3d_1_4_down_1_6]
        if self.project_res_6_12:
            input_1_6s.append(x3d_input_1_6)
#        print("length 1:6", len(input_1_6s))
        x3d_1_6 = self.merge_1_6_enc(input_1_6s)
        x3d_1_6 = self.process_1_6(x3d_1_6)
        x3d_1_6_down_1_8 = self.down_6_8(x3d_1_6)
        input_1_8s = [x3d_1_6_down_1_8]
        if self.skip_level >= 2:
            input_1_8s.append(x3d_1_4_down_1_8)
            x3d_1_6_down_1_12 = self.down_6_8_12(x3d_1_6_down_1_8)
        if self.skip_level >= 3:
            x3d_1_6_down_1_16 = self.down_6_8_12_16(x3d_1_6_down_1_12)

        if self.project_res_8_16:
            input_1_8s.append(x3d_input_1_8)
#        print("length 1:8", len(input_1_8s))
        x3d_1_8 = self.merge_1_8_enc(input_1_8s)
#        x3d_1_8 = self.merge_1_8_enc([x3d_input_1_8, x3d_1_4_down_1_8, x3d_1_6_down_1_8])
        x3d_1_8 = self.process_1_8(x3d_1_8)
#        x3d_1_8_down_1_12 = self.down_8_12(x3d_1_8)
#        input_1_12s = [x3d_1_8_down_1_12]
#        if self.skip_level >= 2:
#            input_1_12s.append(x3d_1_6_down_1_12)
#            x3d_1_8_down_1_16 = self.down_8_12_16(x3d_1_8_down_1_12)
#        if self.skip_level >= 3:
#            input_1_12s.append(x3d_1_4_down_1_12)
#
#        if self.project_res_6_12:
#            input_1_12s.append(x3d_input_1_12)
##        print("length 1:12", len(input_1_12s))
##        x3d_1_12 = self.merge_1_12_enc([x3d_input_1_12, x3d_1_4_down_1_12, x3d_1_6_down_1_12, x3d_1_8_down_1_12])
#        x3d_1_12 = self.merge_1_12_enc(input_1_12s)
#        x3d_1_12 = self.process_1_12(x3d_1_12)
#        x3d_1_12_down_1_16 = self.down_12_16(x3d_1_12)
#
#        input_1_16s = [x3d_1_12_down_1_16]
#        if self.skip_level >= 2:
#            input_1_16s.append(x3d_1_8_down_1_16)
#        if self.skip_level >= 3:
#            input_1_16s.append(x3d_1_6_down_1_16)
#        if self.skip_level >= 4:
#            input_1_16s.append(x3d_1_4_down_1_16)
#
#        if self.project_res_8_16:
#            input_1_16s.append(x3d_input_1_16)
##        print("length 1:16", len(input_1_16s))
#        x3d_1_16 = self.merge_1_16_enc(input_1_16s)
##        x3d_1_16 = self.merge_1_16_enc([x3d_input_1_16, x3d_1_4_down_1_16, x3d_1_6_down_1_16, x3d_1_8_down_1_16, x3d_1_12_down_1_16])
#        x3d_1_16 = self.process_1_16(x3d_1_16)


        if self.context_prior == "CRCP":
            masks_1_8 = input_dict['masks_1_8']
            ret = self.CP_implicit_pairwise(x3d_1_8, masks_1_8, pts_cam_1_8, max_k) 
            x3d_1_8_CP = ret['x'] 
            for k in ret.keys():
                res[k] = ret[k]

#        x3d_up_1_16 = self.process_1_16_dec(x3d_1_16_CP)
#
#        x3d_up_1_12 = self.up_1_16_1_12(x3d_up_1_16)
#        x3d_up_1_12 = self.merge_1_12_dec([x3d_up_1_12, x3d_1_12]) 

#        x3d_up_1_8 = self.up_1_12_1_8(x3d_up_1_12)
        x3d_up_1_8 = self.merge_1_8_dec([x3d_1_8_CP, x3d_1_8]) 

        x3d_up_1_6 = self.up_1_8_1_6(x3d_up_1_8)
        x3d_up_1_6 = self.merge_1_6_dec([x3d_up_1_6, x3d_1_6])

        x3d_up_1_4 = self.up_1_6_1_4(x3d_up_1_6)
        x3d_up_1_4 = self.merge_1_4_dec([x3d_up_1_4, x3d_1_4])
        ssc_logit_1_4 = self.ssc_head_1_4(x3d_up_1_4)
        res['1_4'] = ssc_logit_1_4

        return res


if __name__ == '__main__':
    model = Network(class_num=12, norm_layer=nn.BatchNorm3d, feature=128, eval=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    left = torch.rand(1, 3, 480, 640).cuda()
    right = torch.rand(1, 3, 480, 640).cuda()
    depth_mapping_3d = torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
    tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

    out = model(left, depth_mapping_3d, tsdf, None)
