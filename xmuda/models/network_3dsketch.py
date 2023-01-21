# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xmuda.models.LMSCNet import SegmentationHead
from xmuda.models.context_prior import ContextPrior3D
from xmuda.models.context_prior_v2 import ContextPrior3Dv2
from xmuda.models.CP_baseline import CPBaseline
from xmuda.models.CP_implicit import CPImplicit
from xmuda.models.CP_implicit_pairwise import CPImplicitPairwise
#from xmuda.models.CP_implicit_pairwise_v2 import CPImplicitPairwise
from xmuda.models.CP_implicit_leftnonempty import CPImplicitV2
from xmuda.models.DDR import Bottleneck3D
from functools import partial
from collections import OrderedDict


class Decoder3D(nn.Module):
    def __init__(self, class_num, norm_layer,
                 non_empty_ratio=0.2,
                 max_k=256,
                 context_prior=None,
                 output_resolutions=['1_4'],
                 in_channels={'1_16': 256, '1_8': 128, '1_4': 128},
                 CP_res="1_16",
                 feature=128,
                 bn_momentum=0.1):
        super(Decoder3D, self).__init__()
        self.business_layer = []
        self.CP_res = CP_res

        self.output_resolutions = output_resolutions
        self.in_channels = in_channels
        self.feature = feature

        self.resize_input_1_4 = nn.Conv3d(256 + 3, 128, kernel_size=1)
        self.resize_input_1_8 = nn.Conv3d(512 + 3, 128, kernel_size=1)
        self.resize_input_1_16 = nn.Conv3d(1024 + 3, 256, kernel_size=1)

        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)

        self.down_1_4_1_8 = Bottleneck3D(feature,
                                         feature // 4,
                                         bn_momentum=bn_momentum,
                                         expansion=4, stride=2,
                                         downsample=nn.Sequential(
                                             nn.AvgPool3d(kernel_size=2, stride=2),
                                             nn.Conv3d(feature, feature, kernel_size=1, stride=1, bias=False), 
                                             norm_layer(feature, momentum=bn_momentum),
                                         ), 
                                         norm_layer=norm_layer)

        self.main_1_8 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )

        self.down_1_8_1_16 = Bottleneck3D(feature,
                                          feature // 4,
                                          bn_momentum=bn_momentum,
                                          expansion=8, stride=2,
                                          downsample=nn.Sequential(
                                              nn.AvgPool3d(kernel_size=2, stride=2), 
                                              nn.Conv3d(feature, feature * 2, kernel_size=1, stride=1, bias=False), 
                                              norm_layer(feature * 2, momentum=bn_momentum),), 
                                          norm_layer=norm_layer)

        self.main_1_16 = nn.Sequential(
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )

        self.up_1_16_1_8 = nn.Sequential( 
            nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1), 
            norm_layer(feature, momentum=bn_momentum), 
            nn.ReLU(inplace=False)
        )

        self.up_1_8_1_4 = nn.Sequential( 
            nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1), 
            norm_layer(feature, momentum=bn_momentum), 
            nn.ReLU(inplace=False)
        )

        self.ssc_head_1_4 = nn.Sequential( 
            nn.Dropout3d(.1), 
            SegmentationHead(feature, feature, class_num, [1, 2, 3])
        )

        if '1_8' in self.output_resolutions:
            self.ssc_head_1_8 = nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            )

        if '1_16' in self.output_resolutions:
            self.ssc_head_1_16 = nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature * 2, class_num, kernel_size=1, bias=True)
            )

        self.enc_1_8 = nn.Sequential(
            Bottleneck3D(in_channels['1_8'], in_channels['1_8'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(in_channels['1_8'], in_channels['1_8'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(in_channels['1_8'], in_channels['1_8'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.enc_1_16 = nn.Sequential(
            Bottleneck3D(in_channels['1_16'], in_channels['1_16'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(in_channels['1_16'], in_channels['1_16'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(in_channels['1_16'], in_channels['1_16'] // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.resize_1_8 = nn.Conv3d(feature + in_channels['1_8'], feature, kernel_size=1)
        self.resize_1_8_up = nn.Conv3d(feature * 2, feature, kernel_size=1)
        self.resize_1_16 = nn.Conv3d(feature * 2 + in_channels['1_16'], feature * 2, kernel_size=1)
        self.resize_1_4_up = nn.Conv3d(feature * 2, feature, kernel_size=1)

        self.context_prior = context_prior
        if context_prior == "CRCP":
#            self.CRCP_layer = ContextPrior3D(feature * 2, (15, 9, 15), norm_layer, class_num, bn_momentum)
            self.CP_implicit_pairwise = CPImplicitPairwise(feature * 2, (15, 9, 15), 
                                                           non_empty_ratio=non_empty_ratio,
                                                           max_k=max_k, 
                                                           n_classes=class_num, 
                                                           bn_momentum=bn_momentum)
#            self.CRCP_layer = ContextPrior3D(feature, (30, 18, 30), norm_layer, class_num, bn_momentum)
        elif context_prior == "CPImplicit":
            if self.CP_res == "1_16":
                self.CPImplicit_layer = CPImplicit(feature * 2, (15, 9, 15), 
                                                   non_empty_ratio=non_empty_ratio, 
                                                   max_k=max_k)
#                self.CPImplicit_layer = CPImplicitV2(feature * 2, (15, 9, 15))
            elif self.CP_res == "1_8":
                self.CPImplicit_layer = CPImplicit(feature, (30, 18, 30))
        elif context_prior == 'CP':
            self.CP_layer = CPBaseline(feature * 2, (15, 9, 15), norm_layer, bn_momentum)
#            self.CP_layer = CPBaseline(feature, (30, 18, 30), norm_layer, bn_momentum)
        
#        self.resize_1_16_transformer = nn.Conv3d(feature * 4, feature * 2, kernel_size=1)
            
#        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)        
#        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=6)
#        self.positional_encodings = nn.Parameter(torch.rand(15 * 9 * 15, 256), requires_grad=True)

    def forward(self, input_dict):
        x3d_input_1_4 = self.resize_input_1_4(input_dict['x3d_1_4'])
        x3d_input_1_8 = self.resize_input_1_8(input_dict['x3d_1_8'])
        x3d_input_1_16 = self.resize_input_1_16(input_dict['x3d_1_16'])
        res = {}

        x3d_1_8 = self.down_1_4_1_8(x3d_input_1_4)
        x3d_input_1_8 = self.enc_1_8(x3d_input_1_8)
        x3d_1_8 = torch.cat([x3d_1_8, x3d_input_1_8], dim=1)
        x3d_1_8 = self.resize_1_8(x3d_1_8)
        x3d_1_8 = self.main_1_8(x3d_1_8)

        x3d_1_16 = self.down_1_8_1_16(x3d_1_8)
        x3d_input_1_16 = self.enc_1_16(x3d_input_1_16)
        x3d_1_16 = torch.cat([x3d_1_16, x3d_input_1_16], dim=1)
        x3d_1_16 = self.resize_1_16(x3d_1_16)
        x3d_1_16 = self.main_1_16(x3d_1_16)


        if self.context_prior == 'CP':
            masks_1_16 = input_dict['masks_1_16']
            x3d_1_16, P = self.CP_layer(x3d_1_16, masks_1_16) 
            res['P'] = P

        if self.context_prior == 'CPImplicit':
            if self.CP_res == "1_16":
                masks_1_16 = input_dict['masks_1_16']
                ret = self.CPImplicit_layer(x3d_1_16, masks_1_16) 
                x3d_1_16 = ret['x'] 
                for k in ret.keys():
                    res[k] = ret[k]
#                res["P_logits"] = ret['P_logits']
#                res["topk_indices"] = ret['topk_indices']
#                res["non_empty_logits"] = ret['non_empty_logits']
#                res["topM_indices"] = ret['topM_indices']

        if self.context_prior == "CRCP":
            masks_1_16 = input_dict['masks_1_16']
#            x3d_1_16, P_logit = self.CRCP_layer(x3d_1_16, masks_1_16) 
            ret= self.CP_implicit_pairwise(x3d_1_16, masks_1_16) 
            x3d_1_16 = ret['x'] 
            for k in ret.keys():
                res[k] = ret[k]
#            res['P_logits'] = ret['P_logits']
#            res["topk_indices"] = ret['topk_indices']

        if self.context_prior == "RP":
            RP_map_context_1_16 = input_dict['map_context_1_16']
            RP_map_P_1_16 = input_dict['map_P_1_16']
            x3d_1_16, P_logit = self.RP_layer(x3d_1_16, masks_1_16, RP_map_context_1_16, RP_map_P_1_16) 
            res['P_logit'] = P_logit



#        embedding_1_16 = x3d_1_16.reshape(x3d_1_16.shape[0], x3d_1_16.shape[1], -1) + self.positional_encodings.T.unsqueeze(0)
#        embedding_1_16 = embedding_1_16.permute(2, 0, 1)
#        embedding_1_16 = self.transformer_encoder(embedding_1_16)

#        bs, c, h, w, d = x3d_1_16.shape
#        y = torch.matmul(x3d_1_16.reshape(bs, c, -1).permute(0, 2, 1), embedding_1_16[:256, :, :].permute(0, 2, 1))
#        x3d_1_16 = y.permute(0, 2, 1).view(bs, -1, h, w, d)

#        embedding_1_16 = embedding_1_16.permute(1, 2, 0).reshape(x3d_1_16.shape)
#        x3d_1_16 = torch.cat([x3d_1_16, embedding_1_16], dim=1)
#        x3d_1_16 = self.resize_1_16_transformer(x3d_1_16)

        if '1_16' in self.output_resolutions:
            ssc_logit_1_16 = self.ssc_head_1_16(x3d_1_16)
            res["1_16"] = ssc_logit_1_16

        x3d_up_1_8 = self.up_1_16_1_8(x3d_1_16)
#        x3d_up_1_8 = x3d_up_1_8 + x3d_1_8
        x3d_up_1_8 = torch.cat([x3d_up_1_8, x3d_1_8], dim=1)
        x3d_up_1_8 = self.resize_1_8_up(x3d_up_1_8)

        if self.context_prior == 'CPImplicit':
            if self.CP_res == "1_8":
                masks_1_8 = input_dict['masks_1_8']
                ret = self.CPImplicit_layer(x3d_up_1_8, masks_1_8) 
                x3d_up_1_8 = ret['x'] 
                res["P_logits"] = ret['P_logits']
                res["topk_indices"] = ret['topk_indices']
                res["non_empty_logits"] = ret['non_empty_logits']

        if '1_8' in self.output_resolutions:
            ssc_logit_1_8 = self.ssc_head_1_8(x3d_up_1_8)
            res["1_8"] = ssc_logit_1_8

#        if self.context_prior == 'CP':
#            masks_1_8 = input_dict['masks_1_8']
#            x3d_up_1_8, P = self.CP_layer(x3d_up_1_8, masks_1_8) 
#            res['P'] = P
#
#        if self.context_prior == "CRCP":
#            masks_1_8 = input_dict['masks_1_8']
#            x3d_up_1_8, P_logit = self.CRCP_layer(x3d_up_1_8, masks_1_8) 
#            res['P_logit'] = P_logit

#        if self.context_prior == "RP":
#            RP_map_context_1_16 = input_dict['map_context_1_16']
#            RP_map_P_1_16 = input_dict['map_P_1_16']
#            x3d_1_16, P_logit = self.RP_layer(x3d_1_16, masks_1_16, RP_map_context_1_16, RP_map_P_1_16) 
#            res['P_logit'] = P_logit

        x3d_up_1_4 = self.up_1_8_1_4(x3d_up_1_8)
        x3d_up_1_4 = torch.cat([x3d_up_1_4, x3d_input_1_4], dim=1)
        x3d_up_1_4 = self.resize_1_4_up(x3d_up_1_4)    
        ssc_logit_1_4 = self.ssc_head_1_4(x3d_up_1_4)
        res['1_4'] = ssc_logit_1_4

        return res


if __name__ == '__main__':
    model = Network(class_num=12, norm_layer=nn.BatchNorm3d, feature=128, eval=True)
    # print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    left = torch.rand(1, 3, 480, 640).cuda()
    right = torch.rand(1, 3, 480, 640).cuda()
    depth_mapping_3d = torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
    tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

    out = model(left, depth_mapping_3d, tsdf, None)
