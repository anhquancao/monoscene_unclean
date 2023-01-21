import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D
import numpy as np
import math

class CPImplicit(nn.Module):
    def __init__(self, in_channels, size, non_empty_ratio=0.2, max_k=256, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.non_empty_ratio = non_empty_ratio
        self.max_k = max_k
        print(self.non_empty_ratio, self.max_k)
        self.flatten_size = size[0] * size[1] * size[2]
#        self.resize_input = nn.Conv3d(in_channels, in_channels * 2, kernel_size=1)
#        in_channels *= 2
        feature = in_channels
        self.agg = nn.Sequential(
#            Bottleneck3D(feature, feature // 4, expansion=8, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[3, 3, 3]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[4, 4, 4]),
        )

        self.pred_non_empty_logit = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

#        f = 32 # res_1_16
        f = 32 # res_1_8
        self.pred_P_logit = nn.Sequential(
            nn.Linear(in_channels, f),
#            nn.Linear(in_channels, f),
            nn.BatchNorm1d(f),
            nn.ReLU(),
            nn.Linear(f, f),
            nn.BatchNorm1d(f),
            nn.ReLU(),
            nn.Linear(f, 1)
        )
#        self.resize = nn.Conv3d(in_channels * 3, in_channels // 2, kernel_size=1)
        self.resize = nn.Sequential(
            nn.Conv3d(in_channels * (2 + 2), feature * 2, kernel_size=1),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[3, 3, 3]),
            nn.Conv3d(feature * 2, feature, kernel_size=1),
        )
        self.pe = self.positionalencoding1d(in_channels, self.flatten_size) # flatten_size x in_channels
        self.coords = self.get_abs_coords(self.size[0], self.size[1], self.size[2])
  
    def forward(self, input, masks):
        ret = {}
#        input = self.resize_input(input)
        bs, c, h, w, d = input.shape
        x = self.agg(input)

        non_empty_logits = []
        relate_probs = []
        P_logits = []
        list_topk_indices = []

        x_flatten = x.reshape(bs, c, self.flatten_size) # bs, c, 2025
#        abs_pos = torch.arange(self.flatten_size).type_as(x).long()
#        abs_pos = self.coords.type_as(x).reshape(3, self.flatten_size) 
        x_context = torch.zeros((bs, c * 2, self.flatten_size)).type_as(x)
        for i in range(bs):
            mask = masks[i]
            xi = x_flatten[i, :, mask].T # N, c 
#            indices = abs_pos[mask].reshape(-1, 1)
#            indices = abs_pos[:, mask].T # N, 3
            masked_pe = self.pe[mask, :].type_as(xi)
#            xi_with_indices = torch.cat([xi, indices], dim=1) # N, c + 3
            xi += masked_pe
            
            N, _ = xi.shape
            k = np.minimum(int(self.non_empty_ratio * N), self.max_k) 
#            k = int(self.non_empty_ratio * N)

            non_empty_logit = self.pred_non_empty_logit(xi) # N, 1
            non_empty_logits.append(non_empty_logit)
            non_empty_prob = torch.sigmoid(non_empty_logit)
            relate_probs.append(non_empty_prob.squeeze())
#            non_empty_prob = F.softmax(non_empty_logit)
            
            # top_k_indices: k, 1
            # top_k_features: k, 1
#            _, topk_indices = torch.topk(non_empty_prob, k, dim=0)
            # weighted random sampling
            topk_indices = torch.multinomial(non_empty_prob.squeeze(), k, replacement=True)
            list_topk_indices.append(topk_indices)
            topk_features = xi[topk_indices.squeeze(), :] # k, c

            non_empty_prob = non_empty_prob.squeeze()[topk_indices] # k

            
            xi_repeat = torch.repeat_interleave(xi, k , dim=0) # N * k, c 
#            indices_tiled_old = indices[topk_indices.squeeze(), :].repeat(N, 1) # N * k, 3
#            indices_tiled = indices[topk_indices.squeeze(), :].unsqueeze(0).expand(N, -1, -1).reshape(N * k, -1) # N * k, 3
#            indices_repeat_interleave = torch.repeat_interleave(indices, k, dim=0) # N * k, 3

            masked_pe_tiles = masked_pe[topk_indices.squeeze(), :].unsqueeze(0).expand(N, -1, -1).reshape(N * k, -1) # N * k, in_channels
#            xi_repeat += masked_pe_tiles

#            relative_pos = indices_tiled - indices_repeat_interleave
#            xi_with_indices = torch.cat([xi_repeat, relative_pos], dim=1) # N * k, c + 9
            xi_with_indices = xi_repeat + masked_pe_tiles # N * k, c 
            P_logit = self.pred_P_logit(xi_with_indices) # N * k, 1 
            P_logit = P_logit.reshape(N, k) # N, k
            P_logits.append(P_logit)

#            P = torch.sigmoid(P_logit) * non_empty_prob.unsqueeze(0).expand(N, -1) # N, k
            P = torch.sigmoid(P_logit) # N, k

#            x_intra_context = torch.mm(P, topk_features) / torch.sum(P, dim=1, keepdim=True)  # N, c
#            x_inter_context = torch.mm(1 - P, topk_features) / torch.sum(1 - P, dim=1, keepdim=True) # N, c
            non_empty_prob = non_empty_prob.unsqueeze(0).expand(N, -1)
            P_intra = P * non_empty_prob
            P_inter = (1 - P) * non_empty_prob
            x_intra_context = torch.mm(P_intra, topk_features) / torch.sum(P_intra, dim=1, keepdim=True)  # N, c
            x_inter_context = torch.mm(P_inter, topk_features) / torch.sum(P_inter, dim=1, keepdim=True) # N, c

            x_context[i, :c, mask] = x_intra_context.T
            x_context[i, c:, mask] = x_inter_context.T

        
        x_context = x_context.reshape(bs, 2 * c, self.size[0], self.size[1], self.size[2]) 
#        x = torch.cat([input, x_context], dim=1)
        x = torch.cat([input, x_context, x], dim=1)
        x = self.resize(x)
        ret["non_empty_logits"] = non_empty_logits
        ret["relate_probs"] = relate_probs
        ret["P_logits"] = P_logits
        ret['topk_indices'] = list_topk_indices
        ret["x"] = x

        return ret 
    
    @staticmethod
    def get_abs_coords(dx, dy, dz):
        x_dim = torch.arange(15)
        y_dim = torch.arange(9)
        z_dim = torch.arange(15)

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
