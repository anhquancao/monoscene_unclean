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
from xmuda.models.CP_implicit_leftnonempty import CPImplicitV2
from xmuda.models.DDR import Bottleneck3D
from functools import partial
from collections import OrderedDict


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

class Attend(nn.Module):
    def __init__(self):
        super(Attend, self).__init__()
    def forward(self):
        pass 

class Decoder3D(nn.Module):
    def __init__(self, class_num, norm_layer,
                 scene_sizes,
                 features,
                 project_res=[],
                 max_k=128,
                 context_prior=None,
                 bn_momentum=0.1):
        super(Decoder3D, self).__init__()
        self.business_layer = []
        self.project_res = project_res
        
        self.feature_1_4 = features[0]
        self.feature_1_6 = features[1]
        self.feature_1_8 = features[2]
        self.feature_1_12 = features[3]
        self.feature_1_16 = features[4]

        self.feature_1_16_dec = self.feature_1_16 * 3
        self.feature_1_12_dec = self.feature_1_12 * 2
        self.feature_1_8_dec = self.feature_1_8 * 2
        self.feature_1_6_dec = self.feature_1_6 * 2
        self.feature_1_4_dec = self.feature_1_4 * 2 
        self.size_1_4 = scene_sizes[0]
        self.size_1_6 = scene_sizes[1]
        self.size_1_8 = scene_sizes[2]
        self.size_1_12 = scene_sizes[3]
        self.size_1_16 = scene_sizes[4]

        self.process_1_4 = Process(self.feature_1_4, norm_layer, bn_momentum)
        self.process_1_6 = Process(self.feature_1_6, norm_layer, bn_momentum)
        self.process_1_8 = Process(self.feature_1_8, norm_layer, bn_momentum)
        self.process_1_12 = Process(self.feature_1_12, norm_layer, bn_momentum)
        self.process_1_16 = Process(self.feature_1_16, norm_layer, bn_momentum)


        self.down_4_6 = ResizeInterpolate(self.feature_1_4, self.feature_1_6, self.size_1_6, norm_layer, bn_momentum)
        self.down_6_8 = ResizeInterpolate(self.feature_1_6, self.feature_1_8, self.size_1_8, norm_layer, bn_momentum)
        self.down_8_12 = ResizeInterpolate(self.feature_1_8, self.feature_1_12, self.size_1_12, norm_layer, bn_momentum)
        self.down_12_16 = ResizeInterpolate(self.feature_1_12, self.feature_1_16, self.size_1_16, norm_layer, bn_momentum)

        self.process_1_16_dec = Process(self.feature_1_16_dec, norm_layer, bn_momentum)

        self.up_1_16_1_12 = ResizeInterpolate(self.feature_1_16_dec, self.feature_1_12_dec, self.size_1_12, norm_layer, bn_momentum)
        self.up_1_12_1_8 = ResizeInterpolate(self.feature_1_12_dec, self.feature_1_8_dec, self.size_1_8, norm_layer, bn_momentum)
        self.up_1_8_1_6 = ResizeInterpolate(self.feature_1_8_dec, self.feature_1_6_dec, self.size_1_6, norm_layer, bn_momentum)
        self.up_1_6_1_4 = ResizeInterpolate(self.feature_1_6_dec, self.feature_1_4_dec, self.size_1_4, norm_layer, bn_momentum)

        self.merge_1_4_dec = Merge(self.feature_1_4_dec + self.feature_1_4, self.feature_1_4_dec)
        self.merge_1_6_dec = Merge(self.feature_1_6_dec + self.feature_1_6, self.feature_1_6_dec)
        self.merge_1_8_dec = Merge(self.feature_1_8_dec + self.feature_1_8, self.feature_1_8_dec)
        self.merge_1_12_dec = Merge(self.feature_1_12_dec + self.feature_1_12, self.feature_1_12_dec)

#        self.process_1_8_proj = nn.Sequential(
#            nn.Conv3d(self.feature_1_8, self.feature_1_8, kernel_size=3, stride=1, padding=1), 
#            nn.BatchNorm3d(self.feature_1_8), 
#            nn.ReLU()
#        )

        self.ssc_head_1_4 = nn.Sequential( 
            nn.Dropout3d(.1),
            SegmentationHead(self.feature_1_4_dec, self.feature_1_4_dec, class_num, [1, 2, 3]))

        self.context_prior = context_prior
        if context_prior == "CRCP":
            self.CP_implicit_pairwise = CPImplicitPairwise(self.feature_1_16, self.feature_1_16_dec,
                                                           self.feature_1_16, self.size_1_16,
                                                           max_k=max_k,
                                                           n_classes=class_num,
                                                           bn_momentum=bn_momentum)
        else:
            self.resize_16_CP = nn.Conv3d(self.feature_1_16, self.feature_1_16_dec, kernel_size=1, padding=0)

    def forward(self, input_dict):
        pts_cam_1_16 = input_dict['pts_cam_1_16']
        max_k = input_dict['max_k']
        res = {}
        
        x3d_1_4 = input_dict['x3d_1_1']
#        x3d_1_4 = input_dict['x3d_1_4']
#        x3d_1_4 = x3d_1_4 
        for scale in self.project_res:
            x_input = input_dict['x3d_1_{}'.format(scale)] 
            x3d_1_4 = x3d_1_4 + x_input
#        x3d_1_4 = input_dict['x3d_1_4']
#        for scale in ['4', '8', '2', '1']:
#            x_input = input_dict['x3d_1_{}'.format(scale)]
#            x3d_1_4 = F.relu(x3d_1_4 + self.residual_blocks[scale](x_input))
#
        x3d_1_4 = self.process_1_4(x3d_1_4)
        x3d_1_4_down_1_6 = self.down_4_6(x3d_1_4)

        x3d_1_6 = x3d_1_4_down_1_6
        x3d_1_6 = self.process_1_6(x3d_1_6)
        x3d_1_6_down_1_8 = self.down_6_8(x3d_1_6)

        x3d_1_8 = x3d_1_6_down_1_8
#        x3d_1_8_proj = input_dict['x3d_1_1_2']
#        for scale in self.project_res:
#            x_input = input_dict['x3d_1_{}_2'.format(scale)] 
#            x3d_1_8_proj = x3d_1_8_proj + x_input
#        x3d_1_8 = x3d_1_8 + self.process_1_8_proj(x3d_1_8_proj) 
        x3d_1_8 = self.process_1_8(x3d_1_8)
        x3d_1_8_down_1_12 = self.down_8_12(x3d_1_8)

        x3d_1_12 = x3d_1_8_down_1_12
        x3d_1_12 = self.process_1_12(x3d_1_12)
        x3d_1_12_down_1_16 = self.down_12_16(x3d_1_12)

        x3d_1_16 = x3d_1_12_down_1_16
        x3d_1_16 = self.process_1_16(x3d_1_16)

        if self.context_prior == 'CP':
            masks_1_16 = input_dict['masks_1_16']
            x3d_1_16, P = self.CP_layer(x3d_1_16, masks_1_16) 
            res['P'] = P

        if self.context_prior == 'CPImplicit':
            if self.CP_res == "1_16":
                masks_1_16 = input_dict['masks_1_16']
                ret = self.CPImplicit_layer(x3d_1_16, masks_1_16) 
                x3d_1_16 = ret['x'] 
                for k in ret.keys():
                    res[k] = ret[k]

        if self.context_prior == "CRCP":
            masks_1_16 = input_dict['masks_1_16']
            ret = self.CP_implicit_pairwise(x3d_1_16, masks_1_16, pts_cam_1_16, max_k) 
            x3d_1_16_CP = ret['x'] 
            for k in ret.keys():
                res[k] = ret[k]
        else:
            x3d_1_16_CP = self.resize_16_CP(x3d_1_16)

        x3d_up_1_16 = self.process_1_16_dec(x3d_1_16_CP)

        x3d_up_1_12 = self.up_1_16_1_12(x3d_up_1_16)
        x3d_up_1_12 = self.merge_1_12_dec([x3d_up_1_12, x3d_1_12]) 

        x3d_up_1_8 = self.up_1_12_1_8(x3d_up_1_12)
        x3d_up_1_8 = self.merge_1_8_dec([x3d_up_1_8, x3d_1_8]) 

        x3d_up_1_6 = self.up_1_8_1_6(x3d_up_1_8)
        x3d_up_1_6 = self.merge_1_6_dec([x3d_up_1_6, x3d_1_6])

        x3d_up_1_4 = self.up_1_6_1_4(x3d_up_1_6)
        x3d_up_1_4 = self.merge_1_4_dec([x3d_up_1_4, x3d_1_4])
        ssc_logit_1_4 = self.ssc_head_1_4(x3d_up_1_4)

#        if self.use_class_proportion:
#            bs, n_classes, _, _, _ = ssc_logit_1_4.shape
#            ssc_prob_1_4 = torch.sigmoid(ssc_logit_1_4).reshape(bs, n_classes, -1)
#            normalized_ssc_prob_1_4 = ssc_prob_1_4 / ssc_prob_1_4.sum(dim=2, keepdim=True)
#            class_proportion = input_dict['class_proportion']
#            rescaled_ssc_prob_1_4 = class_proportion.unsqueeze(-1) * normalized_ssc_prob_1_4
#            pixel_normalized_ssc_prob_1_4 = rescaled_ssc_prob_1_4 / rescaled_ssc_prob_1_4.sum(dim=1, keepdim=True)
##            print(pixel_normalized_ssc_prob_1_4.sum(1))
#            res['ssc'] = pixel_normalized_ssc_prob_1_4.reshape(ssc_logit_1_4.shape)
##            print(class_proportion, rescaled_ssc_prob_1_4.sum(2))

        res['ssc'] = ssc_logit_1_4

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
