# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xmuda.models.LMSCNet import SegmentationHead, ASPP
from xmuda.models.context_prior import ContextPrior3D
from xmuda.models.context_prior_v2 import ContextPrior3Dv2
from xmuda.models.CP_baseline import CPBaseline
from xmuda.models.CP_implicit import CPImplicit
from xmuda.models.CP_v6 import CPMegaVoxels
#from xmuda.models.CP_v5 import CPMegaVoxels
from xmuda.models.DDR import Bottleneck3D
from functools import partial
from collections import OrderedDict
from xmuda.models.modules import Process, Upsample, Downsample


class Decoder3D(nn.Module):
    def __init__(self, class_num, norm_layer,
                 scene_sizes,
                 features,
                 corenet_proj=None,
                 agg_k=None,
                 n_relations=4,
                 project_res=[],
                 context_prior=None,
                 bn_momentum=0.1):
        super(Decoder3D, self).__init__()
        self.business_layer = []
        self.project_res = project_res
        self.corenet_proj = corenet_proj
        
        self.feature_1_4 = features[0]
        self.feature_1_8 = features[1]
        self.feature_1_16 = features[2]

        self.feature_1_16_dec = self.feature_1_16
        self.feature_1_8_dec = self.feature_1_8
        self.feature_1_4_dec = self.feature_1_4
        self.size_1_4 = scene_sizes[0]
        self.size_1_8 = scene_sizes[1]
        self.size_1_16 = scene_sizes[2]

        if self.corenet_proj is None:
            self.process_1_4 = nn.Sequential(
                Process(self.feature_1_4, norm_layer, bn_momentum, dilations=[1, 2, 3]),
                Downsample(self.feature_1_4, norm_layer, bn_momentum)
            )
            self.process_1_8 = nn.Sequential(
                Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),
                Downsample(self.feature_1_8, norm_layer, bn_momentum)
            )
            self.up_1_16_1_8 = Upsample(self.feature_1_16_dec, self.feature_1_8_dec, norm_layer, bn_momentum)
            self.up_1_8_1_4 = Upsample(self.feature_1_8_dec, self.feature_1_4_dec, norm_layer, bn_momentum)
            self.ssc_head_1_4 = SegmentationHead(self.feature_1_4_dec, self.feature_1_4_dec, class_num, [1, 2, 3])
        else:
            self.up_1_16_1_8 = Upsample(self.feature_1_16_dec * 2, self.feature_1_8_dec, norm_layer, bn_momentum)
            self.up_1_8_1_4 = Upsample(self.feature_1_8_dec * 2, self.feature_1_4_dec, norm_layer, bn_momentum)
            self.unroll = nn.Sequential(
                nn.ConvTranspose3d(2560, self.feature_1_16, kernel_size=(5, 3, 5), stride=1, padding=0),
                norm_layer(self.feature_1_16, momentum=bn_momentum),
                nn.ReLU(),
                nn.ConvTranspose3d(self.feature_1_16, self.feature_1_16, kernel_size=3, stride=3, padding=0),
                norm_layer(self.feature_1_16, momentum=bn_momentum),
                nn.ReLU(),
            )
            self.ssc_head_1_4 = SegmentationHead(self.feature_1_4_dec * 2, self.feature_1_4_dec, class_num, [1, 2, 3])




        self.context_prior = context_prior
        if context_prior == "CRCP":
            self.CP_mega_voxels = CPMegaVoxels(self.feature_1_16,
                                               self.feature_1_16,
                                               self.size_1_16,
                                               n_relations=n_relations,
                                               bn_momentum=bn_momentum)
#
    def forward(self, input_dict):
        res = {}
        
        if self.corenet_proj is None:
            x3d_1_4 = input_dict['x3d']
            x3d_1_8 = self.process_1_4(x3d_1_4)
            x3d_1_16 = self.process_1_8(x3d_1_8)
        else:
            x3d_1_4 = input_dict[4]
            x3d_1_8 = input_dict[8]
            x3d_1_16 = input_dict[16]

            x3d_global = input_dict['x_global']
            x3d_global_unrolled = self.unroll(x3d_global[:, :, None, None, None])
            x3d_1_16 = torch.cat([x3d_global_unrolled, x3d_1_16], dim=1)

        if self.context_prior == "CRCP":
            ret = self.CP_mega_voxels(x3d_1_16)
            x3d_1_16 = ret['x']
            for k in ret.keys():
                res[k] = ret[k]
        if self.corenet_proj is None:
            x3d_up_1_8 = self.up_1_16_1_8(x3d_1_16) + x3d_1_8
            x3d_up_1_4 = self.up_1_8_1_4(x3d_up_1_8) + x3d_1_4
        else:
            x3d_up_1_8 = torch.cat([self.up_1_16_1_8(x3d_1_16), x3d_1_8], dim=1)
            x3d_up_1_4 = torch.cat([self.up_1_8_1_4(x3d_up_1_8), x3d_1_4], dim=1)

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
