import torch
import torch.nn as nn
import torch.nn.functional as F
from xmuda.models.LMSCNet import ASPP2D
from xmuda.efficientnetv2.effnetv2 import effnetv2_xl, effnetv2_b3

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
#                                  nn.ReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
#                                  nn.ReLU())
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=(concat_with.shape[2], concat_with.shape[3]), mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class ExtractAtResolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExtractAtResolution, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_channels),
#                                  nn.ReLU())
                                  nn.LeakyReLU())

    def forward(self, x1, x2, target_size):
        up_x1 = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=True)
        up_x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=True)
        f = torch.cat([up_x1, up_x2], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, bottleneck_features=640, out_feature=128):
        super(DecoderBN, self).__init__()
        features = int(num_features)


        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
#        self.img_W = 640
#        self.img_H = 480
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 16 
        self.feature_1_1 = features // 32 
        self.feature_1_16 = features // 2 

        self.resize_output_1_1 = nn.Conv2d(self.feature_1_1, self.out_feature_1_1, kernel_size=1)
        self.resize_output_1_2 = nn.Conv2d(self.feature_1_2, self.out_feature_1_2, kernel_size=1)
        self.resize_output_1_4 = nn.Conv2d(self.feature_1_4, self.out_feature_1_4, kernel_size=1)
        self.resize_output_1_8 = nn.Conv2d(self.feature_1_8, self.out_feature_1_8, kernel_size=1)
        self.resize_output_1_16 = nn.Conv2d(self.feature_1_16, self.out_feature_1_16, kernel_size=1)

        self.up16 = UpSampleBN(skip_input=features + 256, output_features=self.feature_1_16)
        self.up8 = UpSampleBN(skip_input=self.feature_1_16 + 96, output_features=self.feature_1_8)
        self.up4 = UpSampleBN(skip_input=self.feature_1_8 + 64, output_features=self.feature_1_4)
        self.up2 = UpSampleBN(skip_input=self.feature_1_4 + 32, output_features=self.feature_1_2)
        self.up1 = UpSampleBN(skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1)


#        self.to_res_1_12 = ExtractAtResolution(self.feature_1_16 + self.feature_1_8, self.feature_1_12) 
#        self.to_res_1_6 = ExtractAtResolution(self.feature_1_4 + self.feature_1_8, self.feature_1_6) 

#        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

#        self.res_1_6 = ExtractAtResolution(() 
#        self.up5 = UpSampleBN(skip_input=features // 16 + 3, output_features=features//16)
#        self.conv3 = nn.Conv2d(features // 16, features // 16, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
#        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_1_32 = self.conv2(features["1:32"])

#        print(x_block3.shape)
#        print(x_block2.shape)
#        print(x_block1.shape)
#        print(x_block0.shape)
#        print(features[0].shape)
        x_1_16 = self.up16(x_1_32, features["1:16"])
        x_1_8 = self.up8(x_1_16, features["1:8"])
        x_1_4 = self.up4(x_1_8, features["1:4"])
        x_1_2 = self.up2(x_1_4, features["1:2"])
        x_1_1 = self.up1(x_1_2, features["1:1"])
#        x_1_12 = self.to_res_1_12(x_1_8, x_1_16, (img_size[0]//12 + 1, img_size[1]//12 + 1))
#        x_1_6 = self.to_res_1_6(x_1_4, x_1_8, (img_size[0]//6 + 1, img_size[1]//6 + 1))
#==========================================================
#        x_d4 = self.up4(x_d3, x_block0)
#        x_d5 = self.up5(x_d4, features[0])
#        out = self.conv3(x_d5)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
#        return [x_d1, x_d2, x_d3, x_d4 x_d5]
#        return [x_d1, x_d2, x_d3, x_d4]
#==========================================================

#        x_d0 = self.conv2(x_block4)
#        x_d1 = self.up1(x_d0, x_block3)
#        x_d2 = self.up2(x_d1, x_block2)
#        x_d3 = self.up3(x_d2, x_block1)
#        x_1_4 = x_d3
#        x_1_8 = x_d2
#        x_1_16 = x_d1
#        x_1_12 = self.to_res_1_12(x_1_8, x_1_16)
#        x_1_6 = self.to_res_1_6(x_1_4, x_1_8)
        return {
#            "x_block4": x_block4,
            "1_1": self.resize_output_1_1(x_1_1),
            "1_2": self.resize_output_1_2(x_1_2),
            "1_4": self.resize_output_1_4(x_1_4),
            "1_8": self.resize_output_1_8(x_1_8),
            "1_16": self.resize_output_1_16(x_1_16),
#            "1_1": x_1_1,
#            "1_2": x_1_2,
#            "1_4": x_1_4,
#            "1_8": x_1_8,
#            "1_16": x_1_16,
        }


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.effnetv2 = effnetv2_xl() 
#        block_modules = self.original_model._modules['blocks']._modules

    def forward(self, x):
        features = self.effnetv2(x)
#        for i, feat in enumerate(features):
#            print(i, feat.shape)
        return {
            "1:1": features[0],
            "1:2": features[5],
            "1:4": features[13],
            "1:8": features[21],
            "1:16": features[61],
            "1:32": features[101]
        }


class UEfficientNetV2Encoder(nn.Module):
    def __init__(self, num_features=2048, out_feature=128):
        super(UEfficientNetV2Encoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = DecoderBN(out_feature=out_feature, num_features=num_features)
#        for param in self.encoder.parameters():
#            param.requires_grad = False

    def forward(self, x, **kwargs):
        unet_out = self.decoder(self.encoder(x), **kwargs)
        return unet_out 

    def get_encoder_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_decoder_params(self):  # lr learning rate
        return self.decoder.parameters()

    @classmethod
    def build(cls, **kwargs):
        m = cls()
        return m


if __name__ == '__main__':
    model = UEfficientNetV2Encoder.build()
    x = torch.rand(2, 3, 480, 640)
    pred = model(x)
    print(pred.shape)
