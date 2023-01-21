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
    def __init__(self, in_channels, out_channels, feature, size, n_classes=12, non_empty_ratio=0.2, max_k=256, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.n_classes = n_classes
        self.n_relations = 4
        self.non_empty_ratio = non_empty_ratio
        self.max_k = max_k
        print(self.non_empty_ratio, self.max_k)
        self.flatten_size = size[0] * size[1] * size[2]
        self.agg = nn.Sequential(
#            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[3, 3, 3]),
            ASPP(feature, [1, 2, 3])
        )
        self.context_feature = feature * 2
        self.mega_context = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(feature, self.context_feature, kernel_size=1, stride=1),
            nn.BatchNorm3d(self.context_feature, momentum=bn_momentum),
            nn.ReLU()
        )
        self.flatten_context_size = (size[0]//2) *  (size[1]//2) * (size[2]//2)

        self.context_prior_logit = nn.Sequential(
#            nn.Conv3d(feature, feature, padding=0, kernel_size=1),
#            nn.BatchNorm3d(feature),
#            nn.ReLU(),
            nn.Conv3d(feature, self.n_relations * self.flatten_context_size, padding=0, kernel_size=1)
        )

        self.resize = nn.Conv3d(self.context_feature * self.n_relations + feature, out_channels, kernel_size=1)
#        self.voxel_grid = create_voxel_position(size[0], size[1], size[2]) / torch.tensor([14, 8, 14]).reshape(3, 1, 1, 1)
#        print(torch.min(self.voxel_grid), torch.max(self.voxel_grid))
#        print(self.voxel_grid[:, 1, 2, 3])
#        print(self.voxel_grid[:, 4, 8, 6])
#        print(self.voxel_grid[:, 14, 0, 14])


    def forward(self, input, mask, completion_mask=None):
        ret = {}
        bs, c, h, w, d = input.shape
        mask = mask.reshape(bs, 1, self.size[0], self.size[1], self.size[2])
        x = self.agg(input)
        x_agg = x

        # get the mega context
        x_mega_context = self.mega_context(x_agg) # bs, 512, 7, 4, 7
        x_mega_context = x_mega_context.reshape(bs, x_mega_context.shape[1], -1) # bs, 512, 196
        x_mega_context = x_mega_context.permute(0, 2, 1) # bs, 196, 512
        
        # get context prior map
#        grid_position = self.voxel_grid.type_as(x_agg).unsqueeze(0).expand(bs, -1, -1, -1, -1)
#        print(x_agg.shape, grid_position.shape)
#        x_agg = torch.cat([x_agg, grid_position], dim=1)
        x_context_prior_logit = self.context_prior_logit(x_agg) # bs, 784, 15, 9, 15
        x_context_prior_logit = x_context_prior_logit.reshape(bs, self.n_relations, self.flatten_context_size, self.flatten_size)
        x_context_prior_logit = x_context_prior_logit.permute(0, 1, 3, 2) # bs, n_relation, 2025, 196
        
#        known_logit = self.pred_known(x_agg)

        x_context_prior = torch.sigmoid(x_context_prior_logit)#.detach()
#        x_context_prior = (x_context_prior > 0.5) * x_context_prior
#        x_context_prior[x_context_prior < 0.5] = 0
#        print(x_context_prior_logit.shape)
#        x_context_prior = F.softmax(x_context_prior_logit, dim=1)

        # compute the context features
        x_context_rels = []
        for rel in range(self.n_relations): 
            x_context_prior_rel = x_context_prior[:, rel, :, :]
            x_context_rel = torch.bmm(x_context_prior_rel, x_mega_context) 
            x_context_rels.append(x_context_rel)
        x_context = torch.cat(x_context_rels, dim=2)
        x_context = x_context.permute(0, 2, 1)
        x_context = x_context.reshape(bs, self.context_feature * self.n_relations, 
                                       self.size[0], self.size[1], self.size[2])
#        print(known_logit.shape, x_context.shape)
#        known_prob = torch.sigmoid(known_logit).detach()
#        x_context = x_context * (known_prob > 0.5)

        x_context = x_context * mask
        if completion_mask is not None:
            x_context = x_context * completion_mask
#        for i in range(4):
#            known = x_context_prior[i, :, mask[i], :].sum(0)
#            unknown = x_context_prior[i, :, ~mask[i], :].sum(0)
#            known_ratio = torch.histc(known, bins=5, min=0, max=1)
#            known_ratio = known_ratio / known_ratio.sum()
#            unknown_ratio = torch.histc(unknown, bins=5, min=0, max=1)
#            unknown_ratio = unknown_ratio / unknown_ratio.sum()
#            print("known_{}".format(i), known_ratio)
#            print("unknown_{}".format(i), unknown_ratio) 

        x = torch.cat([input, x_context], dim=1)
        x = self.resize(x)
#        print(x_context.shape, x.shape)

#        ret["relate_probs"] = relate_probs
#        ret["x_agg"] = x_agg
        ret["P_logits"] = x_context_prior_logit
#        ret["known_logits"] = known_logit
#        ret['topk_indices'] = list_topk_indices
        ret["x"] = x

        return ret
    
    @staticmethod
    def get_abs_coords(dx, dy, dz):
        x_dim = torch.arange(dx)
        y_dim = torch.arange(dy)
        z_dim = torch.arange(dz)

        x = x_dim.reshape(-1, 1, 1)
        x = x.expand(-1, dy, dz)
        y = y_dim.reshape(1, -1, 1)
        y = y.expand(dx, -1, dz)
        z = z_dim.reshape(1, 1, -1)
        z = z.expand(dx, dy, -1)

        coords = torch.stack([x, y, z])
        return coords.float()


    @staticmethod
    def positionalencoding1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe
