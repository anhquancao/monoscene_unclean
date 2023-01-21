import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D

class ContextPrior3Dv2(nn.Module):
    def __init__(self, in_channels, size, norm_layer, n_classes, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.n_context_voxels = 512
        self.flatten_size = size[0] * size[1] * size[2]
        self.n_relation_classes = 67 
        self.pred_context_map = nn.Sequential(
#            nn.Conv2d(in_channels, 128, kernel_size=1),
#            norm_layer(128, momentum=bn_momentum),
#            nn.ReLU(),
            nn.Conv2d(in_channels, self.n_relation_classes , kernel_size=1),
#            nn.Sigmoid()
        )

        feature = in_channels
        self.agg = nn.Sequential(
            nn.Conv3d(feature, feature // 2, kernel_size=1),
            Bottleneck3D(feature // 2, feature // 8, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature // 2, feature // 8, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature // 2, feature // 8, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
#        self.resize = nn.Conv3d(feature * 2, in_channels, kernel_size=1)
        self.resize = nn.Conv3d(256 + 128 + 67 * 128, in_channels, kernel_size=1)
  
    def forward(self, input, masks):
        bs, c, h, w, d = input.shape

        x = self.agg(input) # [bs, c, 15, 9, 5]
        x_flatten = x.reshape(bs, x.shape[1], -1) # [bs, c, 2025] 
        
        x_main = x_flatten.unsqueeze(-1).expand(-1, -1, -1, 2025) # [bs, c, 2025, 512]

#        context_idx = torch.randint(0, 2025, (bs, c, self.n_context_voxels)).as_type(x) # bs, c, 512
#        x_context = torch.gather(x_flatten, dim=2, context_idx) # bs, c, 512 
        x_context = x_flatten.unsqueeze(2).expand(-1, -1, 2025, -1) # [bs, c, 2025, 2025]

        x_mix = torch.cat([x_main, x_context], dim=1) # bs, 2 * c , 2025, 512

        P_logit = self.pred_context_map(x_mix)

        #################
        P = F.softmax(P_logit / 0.03, dim=1) # [bs, 67, 2025, 512]
        
        # [4, 2025, c]
        x_flatten = x_flatten.permute(0, 2, 1)
        
        # P: [bs, 67, 2025, 512]

        x_contexts = []
        P_filtered = torch.zeros_like(P)
        for i in range(bs):
            P_filtered[i, :, masks[i], :] = P[i, :, masks[i], :]
            P_filtered[i, :, :, masks[i]] = P[i, :, :, masks[i]]

            # P_filtered[i]: [67, 2025, 512] 
            # Pi : [67 * 2025, 512]
            Pi = P_filtered[i].reshape(-1, self.flatten_size)

            # x_context: [67 * 2025, 256]
            x_context = torch.mm(Pi, x_flatten[i])

            # x_context: [n_relation_class, 2025, 256]
            x_context = x_context.reshape(self.n_relation_classes, self.flatten_size, -1)

            # [2025, 256 * n_relation_class]
            x_context = x_context.transpose(0, 1).reshape(self.flatten_size, -1)
            x_contexts.append(x_context)

        # [4, 2025, 256 * n_relation_class]
        x_context = torch.stack(x_contexts)

        # [4, 256 * n_relation_class, 2025]
        x_context = x_context.transpose(1, 2)

        x_context = x_context.reshape(bs, -1, h, w, d)

        x = torch.cat([input, x_context, x], dim=1)
        x = self.resize(x)

        return x, P_logit





