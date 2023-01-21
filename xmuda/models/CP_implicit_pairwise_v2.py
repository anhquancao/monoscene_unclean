import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D
import numpy as np
import math

class CPImplicitPairwise(nn.Module):
    def __init__(self, in_channels, size, n_classes=12, non_empty_ratio=0.2, max_k=256, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.n_classes = n_classes
        self.n_relations = 3
        self.non_empty_ratio = non_empty_ratio
        self.max_k = max_k
        print(self.non_empty_ratio, self.max_k)
        self.flatten_size = size[0] * size[1] * size[2]
#        self.resize_input = nn.Conv3d(in_channels, in_channels * 2, kernel_size=1)
#        in_channels *= 2
        feature = in_channels
        self.agg = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[3, 3, 3]),
        )

        self.pred_relate_logit = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

#        f = 32 # res_1_16
        f = 32 # res_1_8
        self.pred_P_logit = nn.Sequential(
            nn.Linear(in_channels, f),
            nn.BatchNorm1d(f),
            nn.ReLU(),
            nn.Linear(f, f),
            nn.BatchNorm1d(f),
            nn.ReLU(),
            nn.Linear(f, self.n_relations)
        )
        self.resize = nn.Conv3d(in_channels * (self.n_relations + 1), in_channels, kernel_size=1)
        self.pe = self.positionalencoding1d(in_channels, self.flatten_size) # flatten_size x in_channels
        self.coords = self.get_abs_coords(self.size[0], self.size[1], self.size[2])
  
    def forward(self, input, masks):
        ret = {}
        bs, c, h, w, d = input.shape
        x = self.agg(input)

        P_logits = []
        list_topk_indices = []

        x_flatten = x.reshape(bs, c, self.flatten_size) # bs, c, 2025
        x_context = torch.zeros((bs, c * self.n_relations, self.flatten_size)).type_as(x)
        for i in range(bs):
            mask = masks[i]
            xi = x_flatten[i, :, mask].T # N, c 
            masked_pe = self.pe[mask, :].type_as(xi)
            xi += masked_pe

            N, _ = xi.shape
            k = np.minimum(int(self.non_empty_ratio * N), self.max_k) 

            xi_repeat = xi.unsqueeze(1).expand(-1, N, -1).reshape(N * N, -1) # N * N, c 
            masked_pe_tiles = masked_pe.unsqueeze(0).expand(N, -1, -1).reshape(N * N, -1) # N * k, in_channels
            relate_logit = self.pred_relate_logit(xi_repeat + masked_pe_tiles) # N * N, 1
            relate_prob = torch.sigmoid(relate_logit).reshape(N, N) # N, N
            
            # top_k_indices: k1
            # top_k_features: k, 1
#            _, topk_indices = torch.topk(non_empty_prob, k, dim=0)
            # weighted random sampling
            relate_prob, topk_indices = torch.topk(relate_prob, k) # (N, k)
            list_topk_indices.append(topk_indices)
            topk_features = torch.gather(xi.unsqueeze(1).expand(-1, N, -1), 1, topk_indices.unsqueeze(-1).expand(-1, -1, c)) # (N, k, c)

#            relate_prob = torch.gather(relate_prob, 1, topk_indices) # (N, k)
            
            xi_repeat = xi.unsqueeze(1).expand(-1, k, -1).reshape(N * k, -1) # N * k, c 
#            masked_pe_tiles = masked_pe[topk_indices.squeeze(), :].unsqueeze(0).expand(N, -1, -1).reshape(N * k, -1) # N * k, in_channels
            masked_pe_tiles = masked_pe.unsqueeze(0).expand(N, -1, -1) # N, N, in_channels
            masked_pe_tiles = torch.gather(masked_pe_tiles, 1, topk_indices.unsqueeze(-1).expand(-1, -1, c)).reshape(N * k, c) 
            xi_with_indices = xi_repeat + masked_pe_tiles # N * k, c 

            P_logit = self.pred_P_logit(xi_with_indices) # N * k, 78 
            P_logits.append(P_logit)

#            P = torch.sigmoid(P_logit) * non_empty_prob.unsqueeze(0).expand(N, -1) # N, k
            P = F.softmax(P_logit, dim=1) # N * k, 78 
            relate_prob = relate_prob.unsqueeze(-1).expand(-1, -1, self.n_relations) # N, k, 78
            P = P.reshape(N, k, -1) * relate_prob # N, k, 78 

            # N, 78, c
            x_context_item = torch.bmm(P.transpose(1, 2), topk_features) / torch.sum(P, dim=1, keepdim=False).unsqueeze(-1) 
#            x_context_item = x_context_item.reshape(self.n_relations, N, -1) # 78, N, c

            x_context[i, :, mask] = x_context_item.permute(1, 2, 0).reshape(-1, N) #.self.n_relations * c, N 

        
        x_context = x_context.reshape(bs, self.n_relations * c, self.size[0], self.size[1], self.size[2]) 
        x = torch.cat([input, x_context], dim=1)
        x = self.resize(x)
#        ret["non_empty_logits"] = non_empty_logits
        ret["P_logits"] = P_logits
        ret['topk_indices'] = list_topk_indices
        ret["x"] = x

        return ret 
    
    @staticmethod
    def get_abs_coords(dx, dy, dz):
        x_dim = torch.arange(dx)
        y_dim = torch.arange(dy)
        z_dim = torch.arange(dz)

        x = x_dim.reshape(-1, 1, 1)
        x = x.expand(-1, 9, 15)
        y = y_dim.reshape(1, -1, 1)
        y = y.expand(15, -1, 15)
        z = z_dim.reshape(1, 1, -1)
        z = z.expand(15, 9, -1)

        coords = torch.stack([x, y, z])
        return coords


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
