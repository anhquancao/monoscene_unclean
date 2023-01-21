import torch.nn as nn
import torch.nn.functional as F
import torch

class SSCNetDecoder(nn.Module):
    '''
    # Class coded from caffe model https://github.com/shurans/sscnet/blob/master/test/demo.txt
    '''

    def __init__(self, class_num, in_channels=64):
        '''
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        '''
        super().__init__()

        self.nbr_classes = class_num

        self.conv1_1 =  nn.Conv3d(in_channels, 16, kernel_size=7, padding=3, stride=2, dilation=1)  # conv(16, 7, 2, 1)

        self.reduction2_1 = nn.Conv3d(16, 32, kernel_size=1, padding=0, stride=1, dilation=1)  # conv(32, 1, 1, 1)

        self.conv2_1 =  nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(32, 3, 1, 1)
        self.conv2_2 = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(32, 3, 1, 1)

        self.pool2 = nn.MaxPool3d(2)  # pooling

        self.reduction3_1 = nn.Conv3d(64, 64, kernel_size=1, padding=0, stride=1, dilation=1)  # conv(64, 1, 1, 1)

        self.conv3_1 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(64, 3, 1, 1)
        self.conv3_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(64, 3, 1, 1)

        self.conv3_3 = nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(64, 3, 1, 1)
        self.conv3_4 = nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(64, 3, 1, 1)

        self.conv3_5 = nn.Conv3d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)  # dilated(64, 3, 1, 2)
        self.conv3_6 = nn.Conv3d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)  # dilated(64, 3, 1, 2)

        self.conv3_7 = nn.Conv3d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)  # dilated(64, 3, 1, 2)
        self.conv3_8 = nn.Conv3d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)  # dilated(64, 3, 1, 2)

        self.ssc_head = nn.Sequential(
            nn.Conv3d(192, 128, kernel_size=1, padding=0, stride=1, dilation=1),  # conv(128, 1, 1, 1)
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=1, padding=0, stride=1, dilation=1),  # conv(128, 1, 1, 1)
            nn.ReLU(inplace=True),
#            nn.Conv3d(128, self.nbr_classes, kernel_size=1, padding=0, stride=1, dilation=1)
            nn.ConvTranspose3d(128, self.nbr_classes, kernel_size=4, padding=0, stride=4)
        )

        self.tsdf_head = nn.Sequential( 
            nn.Conv3d(192, 128, kernel_size=1, padding=0, stride=1, dilation=1),  # conv(128, 1, 1, 1)
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=1, padding=0, stride=1, dilation=1),  # conv(128, 1, 1, 1)
            nn.ReLU(inplace=True),
#            nn.Conv3d(128, 1, kernel_size=1, padding=0, stride=1, dilation=1)
            nn.ConvTranspose3d(128, 1, kernel_size=4, padding=0, stride=4),
        )

    def forward(self, x):

        input = x

        out = F.relu(self.conv1_1(input))
        out_add_1 = self.reduction2_1(out)
        out = F.relu((self.conv2_1(out)))
        out = F.relu(out_add_1 + self.conv2_2(out))

        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out_add_2 = self.reduction3_1(out)
        out = F.relu(out_add_2 + self.conv3_2(out))

        out_add_3 = self.conv3_3(out)
        out = self.conv3_4(F.relu(out_add_3))
        out_res_1 = F.relu(out_add_3 + out)

        out_add_4 = self.conv3_5(out_res_1)
        out = self.conv3_6(F.relu(out_add_4))
        out_res_2 = F.relu(out_add_4 + out)

        out_add_5 = self.conv3_7(out_res_2)
        out = self.conv3_8(F.relu(out_add_5))
        out_res_3 = F.relu(out_add_5 + out)

        out = torch.cat((out_res_3, out_res_2, out_res_1), 1)

        ssc_logit = self.ssc_head(out)
        tsdf = self.tsdf_head(out)
#        out = F.relu(self.conv4_1(out))
#        out = F.relu(self.conv4_2(out))
#
#        out = self.deconv_classes(out)

        # out = out.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]

        return {
            "ssc_logit": ssc_logit,
            "tsdf_1_4": tsdf
        }

    def weights_initializer(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init(self):
        self.apply(self.weights_initializer)

    def get_parameters(self):
        return self.parameters()
