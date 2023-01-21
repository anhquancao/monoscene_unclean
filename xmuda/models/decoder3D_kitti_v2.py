# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xmuda.models.LMSCNet import SegmentationHead
from xmuda.models.CP_baseline import CPBaseline
from xmuda.models.CP_implicit import CPImplicit
from xmuda.models.CP_v5 import CPMegaVoxels
#from xmuda.models.CP_v6 import CPMegaVoxels
from xmuda.models.modules import Process, Upsample, Downsample


class Decoder3DKitti(nn.Module):
    def __init__(self, class_num, norm_layer,
                 full_scene_size,
                 feature,
                 project_scale,
                 project_res=[],
                 n_relations=4,
                 context_prior=None,
                 bn_momentum=0.1):
        super(Decoder3DKitti, self).__init__()
        self.business_layer = []
        self.project_res = project_res
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature
        self.n_relations = n_relations
        

        size_l1 = (int(self.full_scene_size[0]/project_scale),
                   int(self.full_scene_size[1]/project_scale),
                   int(self.full_scene_size[2]/project_scale))
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        dilations = [1, 2, 3]
        self.process_l1 = nn.Sequential(
            Process(self.feature* 2, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature * 2, norm_layer, bn_momentum)
        )
        self.process_l2 = nn.Sequential(
            Process(self.feature * 4, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature * 4, norm_layer, bn_momentum)
        )

        self.up_13_l2 = Upsample(self.feature * 8, self.feature * 4, norm_layer, bn_momentum)
        self.up_12_l1 = Upsample(self.feature * 4, self.feature * 2, norm_layer, bn_momentum)
        self.up_l1_lfull = Upsample(self.feature * 2, self.feature, norm_layer, bn_momentum)

        self.ssc_head = SegmentationHead(self.feature, self.feature, class_num, dilations)

        self.context_prior = context_prior
        if context_prior == "CRCP":
            self.CP_mega_voxels = CPMegaVoxels(self.feature * 8,
                                               self.feature * 8,
                                               size_l3,
                                               n_relations=n_relations,
                                               bn_momentum=bn_momentum)

    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict['x3d']

        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l3 = self.process_l2(x3d_l2)

        if self.context_prior == "CRCP":
            ret = self.CP_mega_voxels(x3d_l3)
            x3d_l3 = ret['x']
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        ssc_logit_full, features = self.ssc_head(x3d_up_lfull)

        res['features'] = features
        res['ssc'] = ssc_logit_full

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
