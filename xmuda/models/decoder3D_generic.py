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
from xmuda.models.CP_v5 import CPMegaVoxels
from xmuda.models.DDR import Bottleneck3D
from functools import partial
from collections import OrderedDict
from xmuda.models.modules import Process, Upsample, Downsample



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
        
#        self.feature_1_4 = 128
#        self.feature_1_8 = 256
#        self.feature_1_16 = 512
#
#        self.feature_1_16_dec = self.feature_1_16
#        self.feature_1_8_dec = self.feature_1_8
#        self.feature_1_4_dec = self.feature_1_4 
#        self.size_1_4 = scene_sizes[0]
#        self.size_1_8 = scene_sizes[2]
#        self.size_1_16 = scene_sizes[4]
        convs = []
        for feature in features:
            self.convs.append(Process(feature, norm_layer, bn_momentum, dilations=[1, 2, 3]))
        self.convs = nn.ModuleList(convs)

        downsamples = []
        for feature in features[:-1]:
            downsamples.append(Downsample(feature, norm_layer, bn_momentum))
        self.downsamples = nn.ModuleList(downsamples)

        self.conv_after_CP = Process(features[-1], norm_layer, bn_momentum, dilations=[1, 2, 3])

        upsamples = []
        for feature in features[::-1][-1]:
            upsamples.append(Upsample(feature, norm_layer, bn_momentum))
        self.upsamples = nn.ModuleList(upsamples)
#        self.up_1_16_1_8 = Upsample(self.feature_1_16_dec, self.feature_1_8_dec, norm_layer, bn_momentum)
#        self.up_1_8_1_4 = Upsample(self.feature_1_8_dec, self.feature_1_4_dec, norm_layer, bn_momentum)

        self.ssc_head = SegmentationHead(features[0], features[0], class_num, [1, 2, 3])

        self.context_prior = context_prior
        if context_prior == "CRCP":
            self.CP_mega_voxels = CPMegaVoxels(features[-1], features[-1],
                                               scene_sizes[-1],
                                               bn_momentum=bn_momentum)

    def forward(self, input_dict):
        res = {}
        
        x3d = input_dict['x3d_1_1']
        for scale in self.project_res:
            x_input = input_dict['x3d_1_{}'.format(scale)] 
            x3d = x3d + x_input
        x3ds = [] 
        for i, conv in enumerate(self.convs):
            x3d = conv(x3d)
            x3d.append(x3d)
            if i < len(self.downsamples):
                x3d = self.downsamples[i](x3d)
        x3d_1_4 = self.process_1_4(x3d_1_4)
        x3d_1_4_down_1_8 = self.down_4_8(x3d_1_4)

        x3d_1_8 = x3d_1_4_down_1_8
        x3d_1_8 = self.process_1_8(x3d_1_8)
        x3d_1_8_down_1_16 = self.down_8_16(x3d_1_8)

        x3d_1_16 = x3d_1_8_down_1_16
        x3d_1_16 = self.process_1_16(x3d_1_16)


        if self.context_prior == "CRCP":
            masks_1_16 = input_dict['masks_1_16']
            ret = self.CP_mega_voxels(x3d_1_16, masks_1_16)
            x3d_1_16_CP = ret['x'] 
            for k in ret.keys():
                res[k] = ret[k]
        else:
            x3d_1_16_CP = self.resize_16_CP(x3d_1_16)

        x3d_up_1_16 = self.process_1_16_dec(x3d_1_16_CP)

        x3d_up_1_8 = self.up_1_16_1_8(x3d_up_1_16) + x3d_1_8

        x3d_up_1_4 = self.up_1_8_1_4(x3d_up_1_8) + x3d_1_4

        ssc_logit_1_4 = self.ssc_head_1_4(x3d_up_1_4)

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
