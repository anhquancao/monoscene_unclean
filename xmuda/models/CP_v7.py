import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D
from xmuda.models.LMSCNet import SegmentationHead, ASPP
import numpy as np
from xmuda.models.modules import Process, Upsample, Downsample
import math
from xmuda.data.utils.preprocess import create_voxel_position


class VoxelRelationPrior(nn.Module):
    def __init__(self, feature, size, 
                 n_relations=5,
                 bn_momentum=0.0003):
        super().__init__()
        assert n_relations == 5, "n_relations must be 5"
        self.size = size
        self.feature = feature
        self.n_relations = n_relations # unknown relation
        print("n_relations", self.n_relations)
        self.flatten_size = size[0] * size[1] * size[2]

        self.project_dim = feature // 4

        self.projects = nn.ModuleList([
            nn.Sequential(
#                nn.Conv3d(self.feature, self.project_dim, padding=0, kernel_size=1),
                nn.Conv3d(self.feature, self.flatten_size, padding=0, kernel_size=1),
            ) for i in range(self.n_relations)
        ])
        self.resize = nn.Sequential(
            nn.Conv3d(self.feature * (self.n_relations - 1) + feature, feature, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(feature),
            nn.ReLU(),
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(feature),
            nn.ReLU(),
        )


    def forward(self, x):
        ret = {}
        bs, c, h, w, d = x.shape

        # get context prior map
        x_rel_logits = []
        for rel in range(self.n_relations):
#            x_projected = self.projects[rel](x) # bs, project_dim, H, W, D
#            x_projected = x_projected.reshape(bs, self.project_dim, -1) # bs, project_dim, N
#
#            x_rel_logit = torch.bmm(x_projected.permute(0, 2, 1), x_projected) # bs, N, N
            x_projected = self.projects[rel](x) # bs, N, H, W, D
            x_rel_logit = x_projected.reshape(bs, self.flatten_size, self.flatten_size) # bs, N, N

#            x_rel_logit = x_projected
            x_rel_logits.append(x_rel_logit.unsqueeze(1)) # bs, 1, N, N

        x_rel_logits = torch.cat(x_rel_logits, dim=1) # bs, 5, N, N
        x_rel_probs = F.softmax(x_rel_logits, dim=1)

        x_contexts = []
        for rel in range(self.n_relations - 1):
            x_rel_prob = x_rel_probs[:, rel, :, :] # bs, N, N
            x_context = torch.bmm(x_rel_prob, x.reshape(bs, self.feature, -1).permute(0, 2, 1)) # bs, N, f
            x_context = x_context.permute(0, 2, 1).unsqueeze(1) # bs, 1, f, N
            x_contexts.append(x_context)
        x_contexts = torch.cat(x_contexts, dim=1) # bs, n_rel - 1, f, N
        x_contexts = x_contexts.reshape(bs, (self.n_relations - 1) * self.feature, h, w, d)

        x = torch.cat([x, x_contexts], dim=1)
        x = self.resize(x)

        ret["P_logits"] = x_rel_logits
        ret["x"] = x

        return ret
