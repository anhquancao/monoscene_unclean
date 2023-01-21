# encoding: utf-8
# https://github.com/charlesCXK/TorchSSC/blob/master/model/sketch.nyu/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict
from models.config_sketch import config
from models.resnet_sketch import get_resnet50
from xmuda.models.projection_layer import Project2Dto3D
# from .resnet import get_resnet50


def group_weight(weight_group, module, lr):
  group_decay = []
  group_no_decay = []
  for m in module.modules():
    if isinstance(m, nn.Linear):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
      or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
      if m.weight is not None:
        group_no_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, nn.Parameter):
      group_decay.append(m)
    # else:
    #     print(m, norm_layer)
  # print(module.modules)
  # print( len(list(module.parameters())) , 'HHHHHHHHHHHHHHHHH',  len(group_decay) + len(
  #    group_no_decay))
  assert len(list(module.parameters())) == len(group_decay) + len(
    group_no_decay)
  weight_group.append(dict(params=group_decay, lr=lr))
  weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
  return weight_group


class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out


'''
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
'''
class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu

'''
Input: 60*36*60 sketch
Latent code: 15*9*15
'''
class CVAE(nn.Module):
    def __init__(self, norm_layer, bn_momentum, latent_size=16):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(self.latent_size, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.mean = nn.Conv3d(self.latent_size, self.latent_size, kernel_size=1, bias=True)      # predict mean.
        self.log_var = nn.Conv3d(self.latent_size, self.latent_size, kernel_size=1, bias=True)     # predict log(var).


        self.decoder_x = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(self.latent_size, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.latent_size*2, self.latent_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(self.latent_size, self.latent_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.Dropout3d(0.1),
            nn.Conv3d(self.latent_size, 2, kernel_size=1, bias=True)
        )

    def forward(self, x, gt=None):
        b, c, h, w, l = x.shape

        if self.training:
            gt = gt.view(b, 1, h, w, l).float()
            for_encoder = torch.cat([x, gt], dim=1)
            enc = self.encoder(for_encoder)
            pred_mean = self.mean(enc)
            pred_log_var = self.log_var(enc)

            decoder_x = self.decoder_x(x)

            out_samples = []
            out_samples_gsnn = []
            for i in range(config.samples):
                std = pred_log_var.mul(0.5).exp_()
                eps = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()
                z1 = eps * std + pred_mean
                z2 = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()

                sketch = self.decoder(torch.cat([decoder_x, z1], dim=1))
                out_samples.append(sketch)

                sketch_gsnn = self.decoder(torch.cat([decoder_x, z2], dim=1))
                out_samples_gsnn.append(sketch_gsnn)

            sketch = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples])
            sketch = torch.mean(sketch, dim=0)
            sketch_gsnn = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples_gsnn])
            sketch_gsnn = torch.mean(sketch_gsnn, dim=0)

            return pred_mean, pred_log_var, sketch_gsnn, sketch
        else:
            out_samples = []
            for i in range(config.samples):
                z = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).cuda()
                decoder_x = self.decoder_x(x)
                out = self.decoder(torch.cat([decoder_x, z], dim=1))
                out_samples.append(out)
            sketch_gsnn = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples])
            sketch_gsnn = torch.mean(sketch_gsnn, dim=0)
            return None, None, sketch_gsnn, None

class STAGE1(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 feature_oper=64,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE1, self).__init__()
        self.business_layer = []

        self.oper1 = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, feature_oper, kernel_size=3, padding=1, bias=False),
            norm_layer(feature_oper, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(feature_oper, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper1)

        self.completion_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
                # nn.ReLU(),
            ), norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.completion_layer1)

        self.completion_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
                # nn.ReLU(),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.completion_layer2)

        self.cvae = CVAE(norm_layer=norm_layer, bn_momentum=bn_momentum, latent_size=config.lantent_size)
        self.business_layer.append(self.cvae)

        self.classify_sketch = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, 2, kernel_size=1, bias=True)
            )]
        )
        self.business_layer.append(self.classify_sketch)

    def forward(self, tsdf, depth_mapping_3d, sketch_gt=None):
        '''
        extract 3D feature
        '''
        tsdf = self.oper1(tsdf)
        completion1 = self.completion_layer1(tsdf)
        completion2 = self.completion_layer2(completion1)

        up_sketch1 = self.classify_sketch[0](completion2)
        up_sketch1 = up_sketch1 + completion1
        up_sketch2 = self.classify_sketch[1](up_sketch1)
        pred_sketch_raw = self.classify_sketch[2](up_sketch2)

        _, pred_sketch_binary = torch.max(pred_sketch_raw, dim=1, keepdim=True)        # (b, 1, 60, 36, 60) binary-voxel sketch
        pred_mean, pred_log_var, pred_sketch_gsnn, pred_sketch= self.cvae(pred_sketch_binary.float(), sketch_gt)

        return pred_sketch_raw, pred_sketch_gsnn, pred_sketch, pred_mean, pred_log_var

class STAGE2(nn.Module):
    def __init__(self, class_num, norm_layer, 
                 full_scene_size,
                 output_scene_size,
                 feature_oper=64,
                 resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE2, self).__init__()
        self.business_layer = []

        if eval:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        self.business_layer.append(self.downsample)

        self.resnet_out = resnet_out
        self.feature = feature
        self.ThreeDinit = ThreeDinit

        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.business_layer.append(self.pooling)

        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),  # feature --> feature*2
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1)

        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2)

        if full_scene_size[0] == output_scene_size[0]:
            self.classify_semantic = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                       output_padding=1),
                    norm_layer(feature, momentum=bn_momentum),
                    nn.ReLU(inplace=False),
                ),
                nn.Sequential(
                    nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                    norm_layer(feature, momentum=bn_momentum),
                    nn.ReLU(inplace=False),
                ),
                nn.Sequential(
                    nn.Dropout3d(.1),
#                    nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
                    nn.ConvTranspose3d(feature, class_num, kernel_size=4, padding=0, stride=4)
                )]
            )
        else:
            self.classify_semantic = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                       output_padding=1),
                    norm_layer(feature, momentum=bn_momentum),
                    nn.ReLU(inplace=False),
                ),
                nn.Sequential(
                    nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                    norm_layer(feature, momentum=bn_momentum),
                    nn.ReLU(inplace=False),
                ),
                nn.Sequential(
                    nn.Dropout3d(.1),
                    nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
                )]
            )
        self.business_layer.append(self.classify_semantic)

        self.oper_sketch = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, feature_oper, kernel_size=3, padding=1, bias=False),
            norm_layer(feature_oper, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(feature_oper, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.oper_sketch_cvae = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, feature_oper, kernel_size=3, padding=1, bias=False),
            norm_layer(feature_oper, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(feature_oper, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper_sketch)
        self.business_layer.append(self.oper_sketch_cvae)
        self.project2d3d = Project2Dto3D(full_scene_size[0]//4, full_scene_size[1]//4, full_scene_size[2]//4)
#        self.project2d3d = Project2Dto3D(output_scene_size[0], output_scene_size[1], output_scene_size[2])

    def forward(self, feature2d, depth_mapping_3d, pred_sketch_raw, pred_sketch_gsnn, full_img_size=None):
        # reduce the channel of 2D feature map
        if self.resnet_out != self.feature:
            feature2d = self.downsample(feature2d)
        if full_img_size is None:
            feature2d = F.interpolate(feature2d, scale_factor=16, mode='bilinear', align_corners=True)
        else:
            feature2d = F.interpolate(feature2d, size=full_img_size, mode='bilinear', align_corners=True)

        '''
        project 2D feature to 3D space
        '''
        b, c, h, w = feature2d.shape
#        feature2d = feature2d.view(b, c, h * w).permute(0, 2, 1)  # b x h*w x c
#
#        zerosVec = torch.zeros(b, 1, c).cuda()  # for voxels that could not be projected from the depth map, we assign them zero vector
#        segVec = torch.cat((feature2d, zerosVec), 1)
#
#        segres = [torch.index_select(segVec[i], 0, depth_mapping_3d[i]) for i in range(b)]
#        segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)  # B, (channel), 60, 36, 60
#        print(feature2d.shape, depth_mapping_3d.shape)
#        print(torch.max(depth_mapping_3d), torch.min(depth_mapping_3d))
        segres = self.project2d3d(feature2d, depth_mapping_3d) # mapping at 1:4 resolution

        '''
        init the 3D feature
        '''
        if self.ThreeDinit:
            pool = self.pooling(segres)

            zero = (segres == 0).float()
            pool = pool * zero
            segres = segres + pool

        '''
        extract 3D feature
        '''
        sketch_proi = self.oper_sketch(pred_sketch_raw)
        sketch_proi_gsnn = self.oper_sketch_cvae(pred_sketch_gsnn)

        seg_fea = segres + sketch_proi + sketch_proi_gsnn
        semantic1 = self.semantic_layer1(seg_fea)
        semantic2 = self.semantic_layer2(semantic1)
        up_sem1 = self.classify_semantic[0](semantic2)
        up_sem1 = up_sem1 + semantic1
        up_sem2 = self.classify_semantic[1](up_sem1)
        pred_semantic = self.classify_semantic[2](up_sem2)

        return pred_semantic, None

'''
main network
'''
class Sketch3DSSC(nn.Module):
    def __init__(self, class_num, base_lr, 
                 full_scene_size,
                 output_scene_size,
                 optimize_everywhere=False,
                 feature_oper=64,
                 resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(Sketch3DSSC, self).__init__()
        self.business_layer = []
        self.full_scene_size = full_scene_size
        self.optimize_everywhere = optimize_everywhere
        print("Sketch3DSSC_optimize_everywhere", self.optimize_everywhere)

        self.nbr_classes = class_num
        self.base_lr = base_lr

        if eval:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=config.bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=nn.BatchNorm2d)
        else:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=config.bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=nn.BatchNorm2d)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.stage1 = STAGE1(class_num, nn.BatchNorm3d, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             feature_oper=feature_oper,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage1.business_layer

        self.stage2 = STAGE2(class_num, nn.BatchNorm3d, full_scene_size, 
                             output_scene_size=output_scene_size,
                             resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             feature_oper=feature_oper,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage2.business_layer

    def forward(self, data):

        rgb = data['img']
        tsdf = data['tsdf'].unsqueeze(1)
        depth_mapping_3d = data['MAPPING_2DRGB_3DGRID'].long()
        sketch_gt = data['3D_SKETCH']

        h, w = rgb.size(2), rgb.size(3)

        feature2d = self.backbone(rgb)

        pred_sketch_raw, pred_sketch_gsnn, pred_sketch, pred_mean, pred_log_var = self.stage1(tsdf, depth_mapping_3d, sketch_gt)
#        print(h, w)
        pred_semantic, _ = self.stage2(feature2d, depth_mapping_3d, pred_sketch_raw, pred_sketch_gsnn, full_img_size=(h, w))

        if self.training:
            return {'pred_semantic': pred_semantic,
                    'pred_sketch_raw': pred_sketch_raw,
                    'pred_sketch_gsnn': pred_sketch_gsnn,
                    'pred_sketch': pred_sketch,
                    'pred_mean': pred_mean,
                    'pred_log_var': pred_log_var}
        return {'pred_semantic': pred_semantic,
                 'pred_sketch_gsnn': pred_sketch_gsnn}

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def weights_initializer(self, feature, conv_init, bn_eps, bn_momentum, **kwargs):
      for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
          conv_init(m.weight, **kwargs)
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
          m.eps = bn_eps
          m.momentum = bn_momentum
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
      return

    def weights_init(self):
      module_list = self.business_layer
      conv_init   = nn.init.kaiming_normal_
      bn_eps = 1e-5   # Forcing, this could be passed through config file
      bn_momentum = 0.1   # Forcing, this could be passed through config file
      if isinstance(module_list, list):
        for feature in module_list:
          self.weights_initializer(feature, conv_init, bn_eps, bn_momentum, mode='fan_in')
      else:
        self.weights_initializer(module_list, conv_init, bn_eps, bn_momentum, mode='fan_in')
      return

    def get_parameters(self):
      params_list = []
      for module in self.business_layer:
        params_list = group_weight(params_list, module, self.base_lr)

      return params_list

    def compute_loss(self, scores, ssc_label, data, class_weights, use_3DSketch_nonempty_mask):

      empty_loss_weight = 1
      cri_weights = torch.ones(self.nbr_classes).type_as(scores['pred_semantic'])

      '''
      semantic loss ---------------------------------------------------------
      '''
      if not self.optimize_everywhere:
          if use_3DSketch_nonempty_mask:
            nonempty = data['nonempty']
          else:
            tsdf = data['tsdf_1_1'].cpu().numpy()
#            nonempty = (tsdf < 0.1) & (tsdf != 0) & (data['ssc_label_1_4'] != 255)
            nonempty = (tsdf < 0.1) & (tsdf != 0) & (ssc_label != 255)

      # Indices at which weight equals 1. The array is flattened for nbr of examples in batch
      if self.optimize_everywhere:
        selectindex = torch.nonzero(torch.ones_like(ssc_label).view(-1)).view(-1)  # Occluded voxels equals 1
        cri_weights[0] = 0.05
      else:  
        selectindex = torch.nonzero(nonempty.view(-1)).view(-1)  # Occluded voxels equals 1
#        cri_weights[0] = 0.05
      criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none', weight=cri_weights)
#      print(selectindex.shape, selectindex_2.shape)

      # Getting labels at indices on which weights equals 1
#      filterLabel = torch.index_select(data['ssc_label_1_4'].view(-1), 0, selectindex)
      filterLabel = torch.index_select(ssc_label.view(-1), 0, selectindex)
      # Selecting indices at which weights equals 1 and flatenning as an array pero class percentage predicted
      filterOutput = torch.index_select(scores['pred_semantic'].permute(0, 2, 3, 4, 1).contiguous().view(-1, self.nbr_classes), 0,
                                        selectindex)
      loss_semantic = criterion(filterOutput, filterLabel.long())
      loss_semantic = torch.mean(loss_semantic)  # TODO: There is an error here.. He shouldn't consider index where label == 255 for the mean
      # torch.sum((criterion(filterOutput, filterLabel.long()))) / torch.sum(filterLabel != 255)

      if not self.training:
        losses = {'total': loss_semantic, 'semantic': loss_semantic}
        return losses

      '''
      sketch loss -----------------------------------------------------------
      '''
      if self.optimize_everywhere:
        selectindex = torch.nonzero(torch.ones_like(data['sketch_1_4']).view(-1)).view(-1)  # Occluded voxels equals 1
      else:  
        selectindex = torch.nonzero(nonempty.view(-1)).view(-1)  # Occluded voxels equals 1
      filter_sketch_gt = torch.index_select(data['sketch_1_4'].view(-1), 0, selectindex)
      filtersketch_raw = torch.index_select(scores['pred_sketch_raw'].permute(0, 2, 3, 4, 1).contiguous().view(-1, 2),
                                            0, selectindex)
      filtersketch = torch.index_select(scores['pred_sketch'].permute(0, 2, 3, 4, 1).contiguous().view(-1, 2),
                                        0, selectindex)
      filtersketchGsnn = torch.index_select(scores['pred_sketch_gsnn'].permute(0, 2, 3, 4, 1).contiguous().view(-1, 2),
                                            0, selectindex)

      # TODO: In the sketches there are not 255 indices, the indices are binary on the sketch.. This ignore 255 is not needed...
      sketch_weights = torch.ones(2).type_as(scores['pred_semantic'])
#      sketch_weights[0] = 0.05
      if self.optimize_everywhere:
        sketch_weights[0] = 0.05
      criterion_sketch = nn.CrossEntropyLoss(ignore_index=255, reduction='none', weight=sketch_weights).cuda()
      loss_sketch = criterion_sketch(filtersketch, filter_sketch_gt.long())
      loss_sketch = torch.mean(loss_sketch) # TODO: There is an error here.. He shouldn't consider index where label == 255 for the mean
      loss_sketch_gsnn = criterion_sketch(filtersketchGsnn, filter_sketch_gt.long())
      loss_sketch_gsnn = torch.mean(loss_sketch_gsnn) # TODO: There is an error here.. He shouldn't consider index where label == 255 for the mean
      loss_sketch_raw = criterion_sketch(filtersketch_raw, filter_sketch_gt.long())
      loss_sketch_raw = torch.mean(loss_sketch_raw) # TODO: There is an error here.. He shouldn't consider index where label == 255 for the mean

      ''' 
      KLD loss --------------------------------------------------------------
      '''
      KLD = -0.5 * torch.mean(1 + scores['pred_log_var'] - scores['pred_mean'].pow(2) - scores['pred_log_var'].exp())

      # TODO: I should do something to have track of each of the losses...
      loss = loss_semantic \
             + (loss_sketch + loss_sketch_raw) * config.sketch_weight \
             + loss_sketch_gsnn * config.sketch_weight_gsnn \
             + KLD * config.kld_weight

      losses = {'total':loss, 'semantic':loss_semantic, 'sketch':loss_sketch, 'sketch_raw':loss_sketch_raw,
                'sketch_gsnn':loss_sketch_gsnn, 'KLD': KLD}

      return losses

    def get_target(self, data):
      '''
      Return the target to use for evaluation of the model
      '''

      return data['3D_LABEL']['1_4']  # .permute(0, 2, 1, 3)

    def get_validation_loss_keys(self):
      return ['total', 'semantic']

    def get_train_loss_keys(self):
      return ['total', 'semantic', 'sketch', 'sketch_raw', 'sketch_gsnn', 'KLD']


if __name__ == '__main__':

  try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
  except ImportError:
    raise ImportError(
      "Please install apex from https://www.github.com/nvidia/apex .")

  print('Starting...')
  model = Sketch3DSSC(class_num=12, feature=128, eval=True)
  # print(model)
  print('model loaded...')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  print('model laoded on device...')
  model.eval()
  print('model passed to eval mode...')

  left = torch.rand(1, 3, 480, 640).cuda()
  right = torch.rand(1, 3, 480, 640).cuda()
  depth_mapping_3d = torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
  tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

  print('model forward pass...')
  out = model(left, depth_mapping_3d, tsdf, None)
  print('model forward pass done...')

