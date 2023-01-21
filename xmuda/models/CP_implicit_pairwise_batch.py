import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D
from xmuda.models.LMSCNet import SegmentationHead
import numpy as np
import math

class MLP(nn.Module):
    def __init__(self, in_channels, feature, dim_coords, dim_project, out_channels, n_layers):
        super().__init__()
        self.project_input = nn.Sequential(
            nn.Linear(dim_coords, dim_project),
            nn.BatchNorm1d(dim_project),
            nn.Linear(dim_project, dim_project),
            nn.BatchNorm1d(dim_project),
            nn.ReLU()
        )
        self.n_layers = n_layers
        self.layers = []
        for layer in range(n_layers):
            in_feature = feature
            if layer == 0:
                in_feature = in_channels
            if layer == n_layers // 2:
                in_feature += dim_project

            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_feature, feature),
                    nn.BatchNorm1d(feature),
                    nn.ReLU()
                )
            )
        self.layers = nn.ModuleList(self.layers)
        self.out = nn.Sequential(
            nn.Linear(feature, out_channels),
        )
    
    def forward(self, x, coords):
        coords = self.project_input(coords)
        for i, layer in enumerate(self.layers):
            if i == self.n_layers // 2:
                x = torch.cat([x, coords], dim=1)
            x = layer(x)
        x = self.out(x)
        return x


class CPImplicitPairwise(nn.Module):
    def __init__(self, in_channels, out_channels, feature, size, n_classes=12, non_empty_ratio=0.2, max_k=256, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.n_classes = n_classes
#        self.n_relations = 3
        self.n_relations = 5
        self.non_empty_ratio = non_empty_ratio
        self.max_k = max_k
        print(self.non_empty_ratio, self.max_k)
        self.flatten_size = size[0] * size[1] * size[2]
        if feature != in_channels:
            self.resize_input = nn.Conv3d(in_channels, feature, kernel_size=1)
        else:
            self.resize_input = nn.Identity() 
        self.agg = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[3, 3, 3]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=nn.BatchNorm3d, dilation=[4, 4, 4]),
        )

        self.pred_relate_logit = MLP(feature, 256, 3, 32, 1, n_layers=4)

#        f = 32 # res_1_16
        self.pred_P_logit = MLP(feature, 256, 6, 64, self.n_relations, n_layers=6)

#        self.resize = nn.Conv3d(in_channels * 3, in_channels // 2, kernel_size=1)
        if (feature * (self.n_relations + 1)) != out_channels:
            self.resize = nn.Conv3d(feature * (self.n_relations + 1), out_channels, kernel_size=1)
        else:
            self.resize = nn.Identity()

        self.pe = self.positionalencoding1d(feature, self.flatten_size) # flatten_size x in_channels
#        self.coords = self.get_abs_coords(self.size[0], self.size[1], self.size[2]).reshape(3, -1)
#        print(self.coords.min(1), self.coords.max(1))
#        self.coords[0, :] = self.coords[0, :] / (size[0]-1 - 0.5)
#        self.coords[1, :] = self.coords[1, :] / 8 - 0.5
#        self.coords[2, :] = self.coords[2, :] / 14 - 0.5
#        print(self.coords.min(1), self.coords.max(1))
#        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=feature, nhead=4)        
#        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=4)

    def forward(self, input, masks, pts_cam, max_k):
        ret = {}
        input = self.resize_input(input)
        bs, c, h, w, d = input.shape
        x = self.agg(input)
        x_agg = x

#        x = input
#        embedding_1_16 = input.reshape(bs, c, -1) + self.pe.type_as(input).T.unsqueeze(0)
#        embedding_1_16 = embedding_1_16.permute(2, 0, 1)
#        x = self.transformer_encoder(embedding_1_16)
#        x = x.permute(1, 2, 0).reshape(input.shape)

        relate_probs = []
        P_logits = []
        list_topk_indices = []

        x_flatten = x.reshape(bs, c, self.flatten_size) # bs, c, 2025
        x_context = torch.zeros((bs, c * self.n_relations, self.flatten_size)).type_as(x)
        for i in range(bs):
            mask = masks[i]
            xi = x_flatten[i, :, mask].T # N, c 
            N, _ = xi.shape
#            indices = abs_pos[mask].reshape(-1, 1)
#            indices = abs_pos[:, mask].T # N, 3
#            abs_coords = self.coords.T[mask, :].type_as(xi)
            abs_coords = pts_cam[i][mask, :].type_as(xi)
            masked_pe = self.pe[mask, :].type_as(xi)
#            xi_with_indices = torch.cat([xi, abs_coords], dim=1) # N, c + 3
#            xi += masked_pe
            
#            k = np.minimum(int(self.non_empty_ratio * N), self.max_k) 
#            k = int(np.minimum(int(non_empty_ratio * N), max_k))
            k = int(np.minimum(N, max_k))
#            k = int(self.non_empty_ratio * N)

#            relate_logit = self.pred_relate_logit(xi_with_indices) # N, 1
            relate_logit = self.pred_relate_logit(xi, abs_coords) # N, 1

            relate_prob = torch.sigmoid(relate_logit)
            relate_probs.append(relate_prob.squeeze())
#            non_empty_prob = F.softmax(non_empty_logit)
            
            # top_k_indices: k1
            # top_k_features: k, 1
#            _, topk_indices = torch.topk(non_empty_prob, k, dim=0)
            # weighted random sampling
            topk_indices = torch.multinomial(relate_prob.squeeze(), k, replacement=True) # (k,)
            list_topk_indices.append(topk_indices)
            topk_features = xi[topk_indices, :] # k, c

            relate_prob = relate_prob[topk_indices, :].squeeze() # k, 1
            
            xi_repeat = xi.unsqueeze(1).expand(-1, k, -1).reshape(N * k, -1) # N * k, c 
#            abs_coord_repeat = abs_coords.unsqueeze(1).expand(-1, k, -1).reshape(N * k, -1) # N * k, c 
            masked_pe_tiles = masked_pe[topk_indices.squeeze(), :].unsqueeze(0).expand(N, -1, -1).reshape(N * k, -1) # N * k, in_channels
            coord_repeat = abs_coords.unsqueeze(1).expand(-1, k, -1).reshape(N * k, -1) # N * k, c 
            coord_tile = abs_coords[topk_indices.squeeze(), :].unsqueeze(0).expand(N, -1, -1).reshape(N * k, -1) # N * k, in_channels
            relative_coord = coord_tile - coord_repeat
#            xi_with_indices = xi_repeat + masked_pe_tiles # N * k, c 
            abs_rel_coords = torch.cat([coord_repeat,relative_coord], dim=1) # N * k, c 

            P_logit = self.pred_P_logit(xi_repeat, abs_rel_coords) # N * k, 78 
            P_logits.append(P_logit)

#            P = torch.sigmoid(P_logit) * non_empty_prob.unsqueeze(0).expand(N, -1) # N, k
            P = F.softmax(P_logit, dim=1) # N * k, 78 
            relate_prob = relate_prob.unsqueeze(0).expand(N * self.n_relations, -1) # 78 * N, k
            P = P.T.reshape(self.n_relations, N, k).reshape(-1, k) * relate_prob # 78 * N, k

            x_context_item = torch.mm(P, topk_features) / torch.sum(P, dim=1, keepdim=True) # 78 * N, c
            x_context_item = x_context_item.reshape(self.n_relations, N, -1) # 78, N, c

            x_context[i, :, mask] = x_context_item.permute(0, 2, 1).reshape(self.n_relations * c, N) 

        
        x_context = x_context.reshape(bs, self.n_relations * c, self.size[0], self.size[1], self.size[2]) 
#        x = torch.cat([input, x_context, x], dim=1)
        x = torch.cat([input, x_context], dim=1)
        x = self.resize(x)


        ret["relate_probs"] = relate_probs
        ret["x_agg"] = x_agg
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
