import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_in = self.conv_classes(x_in)

    return x_in


class LMSCNetEncoder(nn.Module):

    def __init__(self, in_channels=32):
        '''
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        '''

        super().__init__()
        f = 32
        self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

        self.Encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm2d(f),
            nn.ReLU(),
            nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm2d(f),
            nn.ReLU()
        )

        self.Encoder_block2 = nn.Sequential(
          nn.MaxPool2d(2),
          nn.Conv2d(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
          nn.ReLU(),
          nn.Conv2d(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
          nn.ReLU()
        )

        self.Encoder_block3 = nn.Sequential(
          nn.MaxPool2d(2),
          nn.Conv2d(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
          nn.ReLU(),
          nn.Conv2d(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
          nn.ReLU()
        )

        self.Encoder_block4 = nn.Sequential(
          nn.MaxPool2d(2),
          nn.Conv2d(int(f*2), int(f*2.5), kernel_size=3, padding=1, stride=1),
          nn.ReLU(),
          nn.Conv2d(int(f*2.5), int(f*2.5), kernel_size=3, padding=1, stride=1),
          nn.ReLU()
        )

        # Treatment output 1:8
        self.conv_out_scale_1_8 = nn.Conv2d(int(f*2.5), int(f/8), kernel_size=3, padding=1, stride=1)
        self.deconv_1_8__1_2    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=4, padding=0, stride=4)
        self.deconv_1_8__1_1    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=8, padding=0, stride=8)

        # Treatment output 1:4
        self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
        self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)

        self.conv_out_scale_1_4 = nn.Conv2d(int(f*2), int(f/4), kernel_size=3, padding=1, stride=1)    
        
    def forward(self, x):

        input = x
        input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]

        # Encoder block
        _skip_1_1 = self.Encoder_block1(input)
        _skip_1_2 = self.Encoder_block2(_skip_1_1)
        _skip_1_4 = self.Encoder_block3(_skip_1_2)
        _skip_1_8 = self.Encoder_block4(_skip_1_4)

        # Out 1_8
        out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)

        # Out 1_4
        out = self.deconv1_8(out_scale_1_8__2D)
        out = torch.cat((out, _skip_1_4), 1)
        out = F.relu(self.conv1_4(out))
        out_scale_1_4__2D = self.conv_out_scale_1_4(out)

        return out_scale_1_4__2D


class ParallelLMSCNet(nn.Module):

    def __init__(self, 
                 class_num, 
                 class_frequencies, 
                 shared_lmsc_encoder=False,
                 in_channels=32, 
                 n_encoders=4):
        '''
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        '''

        super().__init__()
        self.nbr_classes = class_num
        self.n_encoders = n_encoders
        self.shared_lmsc_encoder = shared_lmsc_encoder
        self.class_frequencies = class_frequencies

        if shared_lmsc_encoder:
            self.encoder = LMSCNetEncoder(in_channels=in_channels) 
        else:
            self.encoders = nn.ModuleList([LMSCNetEncoder(in_channels=in_channels) for _ in range(n_encoders)])

        self.seg_head_1_4 = SegmentationHead(n_encoders, 8, self.nbr_classes, [1, 2, 3])

    def forward(self, xs):
        assert len(xs) == self.n_encoders, "the size of xs should match n_encoders"
        zs = []
        if self.shared_lmsc_encoder:
            for i in range(len(xs)):
                zs.append(self.encoder(xs[i]))
        else:
            for i, encoder in enumerate(self.encoders):
                zs.append(encoder(xs[i]))

        feat = []
        for i, z in enumerate(zs):
            feat.append(z[:, None, :, :, :])

        feat = torch.cat(feat, dim=1)
        out_scale_1_4__3D = self.seg_head_1_4(feat)
        out_scale_1_4__3D = out_scale_1_4__3D.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]

        return out_scale_1_4__3D




