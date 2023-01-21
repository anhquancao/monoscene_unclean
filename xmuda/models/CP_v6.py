import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D
from xmuda.models.LMSCNet import SegmentationHead, ASPP
import numpy as np
from xmuda.models.modules import Process, Upsample, Downsample
import math
from xmuda.data.utils.preprocess import create_voxel_position



class CPMegaVoxels(nn.Module):
    def __init__(self, out_channels, feature, size, 
                 n_relations=4,
                 bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.n_relations = n_relations
        print("n_relations", self.n_relations)
        self.flatten_size = size[0] * size[1] * size[2]
        self.feature = feature
        self.context_feature = feature * 2
        self.flatten_context_size = (size[0]//2) *  (size[1]//2) * (size[2]//2)
        padding = ((size[0] + 1) % 2, (size[1] + 1) % 2, (size[2] + 1) % 2)
        self.mega_context = nn.Sequential(
            nn.Conv3d(feature, self.context_feature, stride=2, padding=padding, kernel_size=3),
        )
        self.flatten_context_size = (size[0]//2) *  (size[1]//2) * (size[2]//2)

        self.context_prior_logits = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.feature, self.flatten_context_size, padding=0, kernel_size=1),
            ) for i in range(n_relations)
        ])
        self.aspp = ASPP(feature, [1, 2, 3])

        self.resize = nn.Sequential(
            nn.Conv3d(self.context_feature * self.n_relations + feature, feature, kernel_size=1, padding=0, bias=False),
            Process(feature, nn.BatchNorm3d, bn_momentum, dilations=[1]),
        )


    def forward(self, input):
        ret = {}
        bs, c, h, w, d = input.shape
#        x_agg = input
        x_agg = self.aspp(input)
        # get the mega context
        x_mega_context_raw = self.mega_context(x_agg) # bs, cf, 7, 4, 7
        x_mega_context = x_mega_context_raw.reshape(bs, self.context_feature, -1) # bs, cf, N/2
        x_mega_context = x_mega_context.permute(0, 2, 1) # bs, N/2, cf

        # get context prior map
        x_context_prior_logits = []
        x_context_rels = []
        for rel in range(self.n_relations):
            x_context_prior_logit = self.context_prior_logits[rel](x_agg) # bs, 784, 15, 9, 15
            x_context_prior_logit = x_context_prior_logit.reshape(bs, self.flatten_context_size, self.flatten_size)
            x_context_prior_logits.append(x_context_prior_logit.unsqueeze(1))

            x_context_prior_logit = x_context_prior_logit.permute(0, 2, 1) # bs, 2025, 196
            x_context_prior = torch.sigmoid(x_context_prior_logit)

            x_context_rel = torch.bmm(x_context_prior, x_mega_context)  # bs, N, f
            x_context_rels.append(x_context_rel)

        x_context = torch.cat(x_context_rels, dim=2)
        x_context = x_context.permute(0, 2, 1)
        x_context = x_context.reshape(bs, x_context.shape[1], self.size[0], self.size[1], self.size[2])

        x = torch.cat([input, x_context], dim=1)
        x = self.resize(x)

        x_context_prior_logits = torch.cat(x_context_prior_logits, dim=1) # bs, n_relations, 196, 2025
        ret["P_logits"] = x_context_prior_logits
        ret["x"] = x

        return ret
