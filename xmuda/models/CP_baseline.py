import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D

class CPBaseline(nn.Module):
    def __init__(self, in_channels, size, norm_layer, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.flatten_size = size[0] * size[1] * size[2]
        self.pred_context_map = nn.Sequential(
            nn.Conv3d(in_channels, self.flatten_size, kernel_size=1),
            norm_layer(self.flatten_size, momentum=bn_momentum),
            nn.Sigmoid()
        )

        feature = in_channels
        self.agg = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.resize = nn.Conv3d(in_channels * 3, in_channels, kernel_size=1)
#        self.positional_encodings = nn.Parameter(torch.rand(in_channels, 
#                                                            size[0], 
#                                                            size[1], 
#                                                            size[2]), requires_grad=True)
  
    def forward(self, input, masks):
        bs, c, h, w, d = input.shape

        x = self.agg(input)
#        x += self.positional_encodings
        P = self.pred_context_map(x)
        P = P.permute(0, 2, 3, 4, 1)
        P = P.reshape(bs, self.flatten_size, self.flatten_size)
        
        x_flatten = x.reshape(bs, c, self.flatten_size).permute(0, 2, 1)
        P_intra = torch.zeros_like(P) 
        P_inter = torch.zeros_like(P) 
        for i in range(bs):
            P_intra[i, masks[i], :] = P[i, masks[i], :]
            P_intra[i, :, masks[i]] = P[i, :, masks[i]]
            P_inter[i, masks[i], :] = 1 - P[i, masks[i], :]
            P_inter[i, :, masks[i]] = 1 - P[i, :, masks[i]]
        x_intra = torch.bmm(P_intra, x_flatten) 
        x_inter = torch.bmm(P_inter, x_flatten) 
       
        x_intra = x_intra.permute(0, 2, 1).reshape(bs, c, h, w, d) 
        x_inter = x_inter.permute(0, 2, 1).reshape(bs, c, h, w, d)

        x = torch.cat([input, x_intra, x_inter], dim=1)
        x = self.resize(x)

        return x, P
