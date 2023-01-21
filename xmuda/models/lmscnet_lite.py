import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class SegmentationHead(nn.Module):
    '''
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    '''

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(
            inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes)
                                  for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes)
                                  for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv3d(
            planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x_in):

        # Dimension exapension
        x_in = x_in[:, None, :, :, :]

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](
            self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i]
                             (self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in


class LMSCNet_SS_lite(nn.Module):

    def __init__(self, class_num, class_frequencies):
        '''
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        '''

        super().__init__()
        self.nbr_classes = class_num
        # Grid dimensions should be (W, H, D).. z or height being axis 1        
        self.class_frequencies = class_frequencies
       
        self.seg_head_1_1 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

    def forward(self, x):

        # input = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
        input = x
        # print("input", input.shape)
        # Reshaping to the right way for 2D convs [bs, H, W, D]
        input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)
        # input = F.max_pool2d(input, 4)
        # print("input_1", input.shape)

       
        out_scale_1_1__3D = self.seg_head_1_1(input)

        # Take back to [W, H, D] axis order
        out_scale_1_1__3D = out_scale_1_1__3D.permute(
            0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]

        # scores = {'pred_semantic_1_1': out_scale_1_1__3D}

        return out_scale_1_1__3D

    def weights_initializer(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init(self):
        self.apply(self.weights_initializer)

    def get_parameters(self):
        return self.parameters()

    def compute_loss(self, pred, target):
        '''
        :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
        '''

        # target = data['3D_LABEL']['1_1']
        device, dtype = target.device, target.dtype
        class_weights = self.get_class_weights().to(
            device=target.device, dtype=target.dtype)
        
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=255, reduction='mean')

        loss = criterion(pred, target.long())        
        
        return loss

    def get_class_weights(self):
        '''
        Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        '''
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(
            1 / np.log(self.class_frequencies + epsilon_w))

        return weights

    def get_target(self, data):
        '''
        Return the target to use for evaluation of the model
        '''
        return {'1_1': data['3D_LABEL']['1_1']}
        # return data['3D_LABEL']['1_1'] #.permute(0, 2, 1, 3)

    def get_scales(self):
        '''
        Return scales needed to train the model
        '''
        scales = ['1_1']
        return scales

    def get_validation_loss_keys(self):
        return ['total', 'semantic_1_1']

    def get_train_loss_keys(self):
        return ['total', 'semantic_1_1']
