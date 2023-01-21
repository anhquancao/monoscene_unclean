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
                 agg_k=None,
                 n_relations=4,
                 project_res=[],
                 context_prior=None,
                 bn_momentum=0.1):
        super(Decoder3D, self).__init__()
        self.business_layer = []
        self.project_res = project_res
        
        self.feature_1_4 = features[0]
        self.feature_1_8 = features[1]

        self.feature_1_8_dec = self.feature_1_8
        self.feature_1_4_dec = self.feature_1_4 
        self.size_1_4 = scene_sizes[0]
        self.size_1_8 = scene_sizes[1]

#        self.completion_head = CompletionHead(norm_layer, bn_momentum)

        self.process_1_4 = nn.Sequential(
            Process(self.feature_1_4, norm_layer, bn_momentum, dilations=[1, 2, 3]),
        )
        self.process_1_8 = nn.Sequential(
            Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),
        )

        self.down_4_8 = Downsample(self.feature_1_4, norm_layer, bn_momentum)


        self.up_1_8_1_4 = Upsample(self.feature_1_8_dec, self.feature_1_4_dec, norm_layer, bn_momentum)

        self.ssc_head_1_4 = SegmentationHead(self.feature_1_4_dec, self.feature_1_4_dec, class_num, [1, 2, 3])

        self.context_prior = context_prior
        print("context_prior", self.context_prior)
        if context_prior == "CRCP":
            self.CP_mega_voxels = CPMegaVoxels(self.feature_1_8,
                                               self.feature_1_8,
                                               self.size_1_8,
                                               n_relations=n_relations,
                                               bn_momentum=bn_momentum)
#
    def forward(self, input_dict):
        res = {}
        
        x3d_1_4 = input_dict['x3d']

        x3d_1_4 = self.process_1_4(x3d_1_4)
        x3d_1_4_down_1_8 = self.down_4_8(x3d_1_4)

        x3d_1_8 = x3d_1_4_down_1_8
        x3d_1_8 = self.process_1_8(x3d_1_8)

        if self.context_prior == "CRCP":
            ret = self.CP_mega_voxels(x3d_1_8)
            x3d_1_8_CP = ret['x']
            for k in ret.keys():
                res[k] = ret[k]
        else:
            x3d_1_8_CP = x3d_1_8

#        x3d_up_1_16 = self.process_1_16_dec(x3d_1_16_CP)

#        x3d_up_1_8 = self.up_1_16_1_8(x3d_up_1_16) + x3d_1_8

        x3d_up_1_4 = self.up_1_8_1_4(x3d_1_8_CP) + x3d_1_4

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
