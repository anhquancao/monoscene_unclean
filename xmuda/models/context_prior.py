import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.DDR import Bottleneck3D

class ContextPrior3D(nn.Module):
    def __init__(self, in_channels, size, norm_layer, n_classes, bn_momentum=0.0003):
        super().__init__()
        self.size = size
        self.flatten_size = size[0] * size[1] * size[2]
        self.n_relation_classes = 12 
#        self.enc = nn.Sequential(
#            nn.Conv3d(in_channels, 4 * self.flatten_size, kernel_size=1),
#            norm_layer(4 * self.flatten_size, momentum=bn_momentum),
#            nn.ReLU()
#        )
#        self.pred_context_map = nn.Sequential(
#            nn.Conv2d(7, 8, kernel_size=1),
#            nn.BatchNorm2d(8),
#            nn.Conv2d(8, n_classes, kernel_size=1),
#        )
        self.pred_context_map = nn.Sequential(
            nn.Conv3d(in_channels, self.flatten_size * self.n_relation_classes , kernel_size=1),
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
        self.resize = nn.Conv3d(feature * (self.n_relation_classes + 2), in_channels, kernel_size=1)
#        self.combine_pred = nn.Conv2d(self.flatten_size, 1, kernel_size=1)

#        self.relative_direction = nn.Parameter(self.compute_relative_direction(size[0], size[1], size[2]), requires_grad=False)
#        self.positional_encodings = nn.Parameter(torch.rand(in_channels, 
#                                                            size[0], 
#                                                            size[1], 
#                                                            size[2]), requires_grad=True)

    @staticmethod 
    def compute_relative_direction(dx, dy, dz):
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

        left = coords.view(3, -1, 1)
        right = coords.view(3, 1, -1)

        relative_direction = left - right
        return relative_direction.unsqueeze(0)


    def forward(self, input, masks):
        bs, c, h, w, d = input.shape

        x = self.agg(input)

##        x += self.positional_encodings
        P_logit = self.pred_context_map(x) # [4, 291600, 15, 9, 15]
        P_logit = P_logit.reshape(bs, self.n_relation_classes, self.flatten_size, h, w, d) # [4, 144, 2025, 15, 9, 15] 

        P_logit = P_logit.reshape(bs, self.n_relation_classes, self.flatten_size, self.flatten_size)
        P = F.softmax(P_logit / 0.03, dim=1)


#        x_enc = self.enc(x).reshape(bs, 4, self.flatten_size, self.flatten_size)
#        x_enc = torch.cat([x_enc, self.relative_direction.expand(bs, -1, -1, -1)], dim=1)
#        P_logit = self.pred_context_map(x_enc)
#        P = F.softmax(P_logit / 0.03, dim=1)

#        group_prediction = P.transpose(1, 2) # (bs, flatten_size, n_relation_classes, flatten_size)
#        group_prediction = self.combine_pred(group_prediction).reshape(bs, self.n_relation_classes, self.flatten_size)

        # [4, 2025, 2025, 67]
        P = P.reshape(bs, self.n_relation_classes, self.flatten_size, self.flatten_size).permute(0, 2, 3, 1)
        
        # [4, 2025, 256]
        x_flatten = x.reshape(bs, c, self.flatten_size).permute(0, 2, 1)

        x_contexts = []
        P_filtered = torch.zeros_like(P)
        for i in range(bs):
            P_filtered[i, masks[i], :, :] = P[i, masks[i], :, :]
            P_filtered[i, :, masks[i], :] = P[i, :, masks[i], :]

            # P_filtered[i]: [2025, 2025, 67 - 1] don't take into account relation with empty space 
            # Pi : [2025 * n_relation_classes, 2025]
            Pi = P_filtered[i, :, :, :].transpose(1, 2).reshape(-1, self.flatten_size)

            # x_context: [2025 * n_relation_classes, 256]
            x_context = torch.mm(Pi, x_flatten[i])

            # x_context: [2025, n_relation_class, 256]
            x_context = x_context.reshape(self.flatten_size, self.n_relation_classes, -1)
#            Pi_total = Pi.reshape(self.flatten_size, self.n_relation_classes, self.flatten_size).sum(dim=2, keepdim=True) + 1e-8 
#            print(x_context.shape, Pi_total.shape)

            # [2025, 256 * (n_relation_class)]
            x_context = x_context.reshape(self.flatten_size, -1)
            x_contexts.append(x_context)

        # [bs, 2025, 256 * (n_relation_class)]
        x_context = torch.stack(x_contexts)

        # [bs, 256 * (n_relation_class - 1), 2025]
        x_context = x_context.transpose(1, 2)

        x_context = x_context.reshape(bs, -1, h, w, d)
#        group_prediction = P.reshape(bs, -1, h, w, d)
        x = torch.cat([input, x_context, x], dim=1)
        x = self.resize(x)

        return x, P_logit





