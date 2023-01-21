# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xmuda.models.LMSCNet import SegmentationHead
from xmuda.models.CP_v6 import CPMegaVoxels
#from xmuda.models.CP_v5 import CPMegaVoxels
from xmuda.models.modules import Process, Upsample, Downsample


class Decoder3DNYU(nn.Module):
    def __init__(self, class_num, norm_layer,
                 scene_sizes,
                 features,
                 corenet_proj=None,
                 agg_k=None,
                 n_relations=4,
                 project_res=[],
                 context_prior=None,
                 bn_momentum=0.1):
        super(Decoder3DNYU, self).__init__()
        self.business_layer = []
        self.project_res = project_res
        self.corenet_proj = corenet_proj
        
        self.feature_1_4 = features[0]
        self.feature_1_8 =  self.feature_1_4 * 2
        self.feature_1_16 = self.feature_1_4 * 4

        self.size_1_4 = scene_sizes[0]
        self.size_1_8 = scene_sizes[1]
        self.size_1_16 = scene_sizes[2]

        self.process_1_4 = nn.Sequential(
            Process(self.feature_1_4, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            # Process(self.feature_1_4, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature_1_4, norm_layer, bn_momentum)
        )
        self.process_1_8 = nn.Sequential(
            Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),            
            Downsample(self.feature_1_8, norm_layer, bn_momentum)
        )
        self.up_1_16_1_8 = Upsample(self.feature_1_16, self.feature_1_8, norm_layer, bn_momentum)
        self.up_1_8_1_4 = Upsample(self.feature_1_8, self.feature_1_4, norm_layer, bn_momentum)
        self.ssc_head_1_4 = SegmentationHead(self.feature_1_4, self.feature_1_4, class_num, [1, 2, 3])

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

        x3d_1_4 = input_dict['x3d']
        x3d_1_8 = self.process_1_4(x3d_1_4)
        x3d_1_16 = self.process_1_8(x3d_1_8)

        if self.context_prior == "CRCP":
            ret = self.CP_mega_voxels(x3d_1_16)
            x3d_1_16 = ret['x']
            for k in ret.keys():
                res[k] = ret[k]
        x3d_up_1_8 = self.up_1_16_1_8(x3d_1_16) + x3d_1_8
        x3d_up_1_4 = self.up_1_8_1_4(x3d_up_1_8) + x3d_1_4

        ssc_logit_1_4, features = self.ssc_head_1_4(x3d_up_1_4)

        res['ssc'] = ssc_logit_1_4
        res['features'] = features

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
