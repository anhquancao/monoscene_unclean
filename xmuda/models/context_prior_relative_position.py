import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D

class RPContextPrior(nn.Module):
    def __init__(self, in_channels, size, norm_layer, n_classes, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.flatten_size = size[0] * size[1] * size[2]
        self.n_relation_classes = 12 
        self.n_relative_positions = 8
        self.pred_context_map = nn.Sequential(
            nn.Conv3d(in_channels, self.n_relative_positions * self.n_relation_classes , kernel_size=1),
#            norm_layer(self.flatten_size, momentum=bn_momentum),
#            nn.Sigmoid()
        )

        feature = in_channels
        self.agg = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
#        self.resize = nn.Conv3d(feature * 2, in_channels, kernel_size=1)
        self.resize = nn.Conv3d(608, in_channels, kernel_size=1)
#        self.combine_pred = nn.Conv2d(self.flatten_size, 1, kernel_size=1)

  
    def forward(self, input, masks, map_contexts, map_Ps):
        bs, c, h, w, d = input.shape

        x = self.agg(input)
        P_logit = self.pred_context_map(x) # [4, 12 * 8, 15, 9, 15]
        P_logit = P_logit.reshape(bs, self.n_relation_classes, self.n_relative_positions, h, w, d) # [4, 12, 8, 15, 9, 15] 

        # 4, 12, 8, 2025
        P_logit = P_logit.reshape(bs, self.n_relation_classes, self.n_relative_positions, self.flatten_size)
        P = F.softmax(P_logit / 0.03, dim=1).reshape(bs, self.n_relation_classes, self.n_relative_positions, h, w, d)

        x_context = torch.zeros((bs,  self.n_relation_classes, self.n_relative_positions, h, w, d)).type_as(x)
        for i in range(bs):
            # 12, 8, 2025
            map_context = map_contexts[i]
            map_P = map_Ps[i]
            if map_context is not None:
                x_context[i, :, map_context[:, 3], map_context[:, 0], map_context[:, 1], map_context[:, 2]] = P[i, :, 
                                                                                                                map_P[:, 3], 
                                                                                                                map_P[:, 0], 
                                                                                                                map_P[:, 1], 
                                                                                                                map_P[:, 2]]

        x_context = x_context.reshape(bs, self.n_relation_classes * self.n_relative_positions, h, w, d)
        x = torch.cat([input, x_context, x], dim=1)
#        x = torch.cat([input, x_context], dim=1)
        x = self.resize(x)

        return x, P_logit





