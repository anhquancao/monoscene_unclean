import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D
from xmuda.models.LMSCNet import SegmentationHead, ASPP
import numpy as np
from xmuda.models.modules import Process, Upsample, Downsample
import math
from xmuda.data.utils.preprocess import create_voxel_position


class AggregationModule(nn.Module):
    """Aggregation Module"""

    def __init__(self,
                 feature, out_feature):
        super(AggregationModule, self).__init__()
        dilations = [1, 2, 3] # kitti
#        dilations = [1, 1, 1] # NYU
        self.b1 = Bottleneck3D(feature, feature // 4, norm_layer=nn.BatchNorm3d, dilation=[dilations[0], dilations[0], dilations[0]])
        self.b2 = Bottleneck3D(feature, feature // 4, norm_layer=nn.BatchNorm3d, dilation=[dilations[1], dilations[1], dilations[1]])
        self.b3 = Bottleneck3D(feature, feature // 4, norm_layer=nn.BatchNorm3d, dilation=[dilations[2], dilations[2], dilations[2]]) 
        self.resize = nn.Conv3d(feature * 4, out_feature, kernel_size=1, padding=0)
        self.aspp = ASPP(out_feature, [1, 2, 3])

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.aspp(self.resize(x))
        return x



class CPMegaVoxels(nn.Module):
    def __init__(self, out_channels, feature, size, 
                 n_relations=4,
                 bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.n_relations = n_relations
        print("n_relations", self.n_relations)
        self.flatten_size = size[0] * size[1] * size[2]
        self.context_feature = feature
        self.agg = AggregationModule(feature, self.context_feature)
        self.mega_context = nn.AvgPool3d(kernel_size=2, stride=2)
        self.flatten_context_size = (size[0]//2) *  (size[1]//2) * (size[2]//2)

        self.context_prior_logits = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.context_feature, self.flatten_context_size, padding=0, kernel_size=1),
            ) for i in range(n_relations)
        ])
        self.resize = nn.Sequential(
            nn.Conv3d(self.context_feature * self.n_relations + feature, out_channels, kernel_size=3, padding=1),
        )

        self.mega_context_logit = nn.Sequential(
            nn.Conv3d(self.context_feature, 12, kernel_size=1, padding=0)
        )


    def forward(self, input):
        ret = {}
        bs, c, h, w, d = input.shape
        x_agg = self.agg(input)

        # get the mega context
        x_mega_context = self.mega_context(x_agg) # bs, 512, 7, 4, 7
        x_mega_context = x_mega_context.reshape(bs, x_mega_context.shape[1], -1) # bs, 512, 196
        x_mega_context = x_mega_context.permute(0, 2, 1) # bs, 196, 512
        
        # get context prior map
        x_context_prior_logits = []
        x_context_rels = []
        for rel in range(self.n_relations):
            x_context_prior_logit = self.context_prior_logits[rel](x_agg) # bs, 784, 15, 9, 15
            x_context_prior_logit = x_context_prior_logit.reshape(bs, 1, self.flatten_context_size, self.flatten_size)
            x_context_prior_logits.append(x_context_prior_logit)

            x_context_prior = torch.sigmoid(x_context_prior_logit).squeeze(dim=1).permute(0, 2, 1) # bs, 2025, 196

            x_context_rel = torch.bmm(x_context_prior, x_mega_context)  # bs, 2025, 1024
            x_context_rels.append(x_context_rel)

        x_context = torch.cat(x_context_rels, dim=2)
        x_context = x_context.permute(0, 2, 1)
        x_context = x_context.reshape(bs, self.context_feature * self.n_relations, self.size[0], self.size[1], self.size[2])


        x = torch.cat([input, x_context], dim=1)
        x = self.resize(x)

        x_context_prior_logits = torch.cat(x_context_prior_logits, dim=1) # bs, n_relations, 196, 2025
        ret["P_logits"] = x_context_prior_logits
        ret["x"] = x

        return ret
