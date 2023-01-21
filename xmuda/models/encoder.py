import torch
import torch.nn as nn
from xmuda.models.resnet34_unet import UNetResNet34
# from xmuda.models.scn_unet import UNetSCN
from xmuda.models.minkunet import MinkUNet34C
import MinkowskiEngine as ME


class Encoder(nn.Module):
    def __init__(self, 
                 edge_extractor=None, 
                 seg_2d=False,
                 edge_rgb_post_process=None,
                 num_classes=20, 
                 num_depth_classes=16, 
                 n_lmscnet_encoders=4):
        super(Encoder, self).__init__()
        assert edge_extractor in ['resunet', 'minkunet', None], "invalid option for edge_extractor"
        assert edge_rgb_post_process in [None, 'conv2d'], "invalid option for edge_rgb_post_process"
        self.edge_extractor = edge_extractor
        self.edge_rgb_post_process = edge_rgb_post_process
        self.seg_2d = seg_2d
        self.num_depth_classes = num_depth_classes
        self.n_lmscnet_encoders = n_lmscnet_encoders
        
        # feature extractor for rgb image
        self.net_rgb = UNetResNet34(input_size=3)

        # feature extractor for edge image
        if self.edge_extractor == 'resunet': 
            self.net_edge = UNetResNet34(pretrained=False, input_size=1)
            feat_channels = 128
        elif self.edge_extractor == 'minkunet':
            self.net_edge = MinkUNet34C(in_channels=1, out_channels=64, D=2)
            feat_channels = 128
        else:
            self.net_edge = nn.Identity()
            feat_channels = 65

        # mix information form rgb and edge extractors
        if self.edge_rgb_post_process == 'conv2d':
            self.net_mix = self.make_conv(feat_channels, [128, 128, 256, 256], [3, 3, 3, 3], [1, 1, 1, 1]) 
            feat_channels = 256
        else:
            self.net_mix = nn.Identity()

        # segmentation head
        if self.seg_2d: 
            self.net_seg_2d = nn.Linear(feat_channels, num_classes)

        self.feat_depth = self.make_conv(feat_channels, 
                                         [128, 128, 256, num_depth_classes * n_lmscnet_encoders], 
                                         [3, 3, 3, 1], 
                                         [1, 1, 1, 0])
        self.depth = self.make_conv(feat_channels, 
                                    [128, 128, 256, num_depth_classes], 
                                    [3, 3, 3, 1], 
                                    [1, 1, 1, 0])
#        self.feat_depth = nn.Sequential(
#            nn.Conv2d(feat_channels, 128, kernel_size=1, padding=0),
#            nn.BatchNorm2d(128),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(128, num_depth_classes, kernel_size=1)
#        )
#
#        self.depth = nn.Sequential(
#            nn.Conv2d(feat_channels, 128, kernel_size=1, padding=0),
#            nn.BatchNorm2d(128),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(128, num_depth_classes, kernel_size=1)
#        )
    
    def make_conv(self, in_channels, layers, kernel_sizes, paddings):
        cur = in_channels 
        modules = []
        for i, n_channels in enumerate(layers[:-1]):
            modules.append(
                nn.Conv2d(cur, n_channels, 
                          kernel_size=kernel_sizes[i], 
                          padding=paddings[i]))
            modules.append(nn.BatchNorm2d(n_channels))
            modules.append(nn.ReLU(inplace=True))
            cur = n_channels

        modules.append(nn.Conv2d(layers[-2], layers[-1],
                                 kernel_size=kernel_sizes[-1],
                                 padding=paddings[-1]))

        return nn.Sequential(*modules)

    def forward(self, img, img_indices, edge, edge_sparse_coord, edge_sparse_feat):
        bs = img.shape[0]
        # 2D network
        x_rgb = self.net_rgb(img)

        if self.edge_extractor == "resunet":
            x_edge = self.net_edge(edge)
        elif self.edge_extractor == "minkunet":
            edge_sinput = ME.SparseTensor(coordinates=edge_sparse_coord, 
                                          features=edge_sparse_feat)
            soutput = self.net_edge(edge_sinput)
            x_edge = torch.zeros_like(x_rgb)
            for i in range(bs):
                coords = soutput.coordinates_at(i).long()
                feats = soutput.features_at(i)
                x_edge[i, :, coords[:, 0], coords[:, 1]] = feats.transpose(0, 1)
        else:
            x_edge = edge

        x = torch.cat([x_rgb, x_edge], dim=1)

        x = self.net_mix(x)

        feat_depth = self.feat_depth(x)
        feat_depth = feat_depth.reshape(bs, self.n_lmscnet_encoders, self.num_depth_classes, feat_depth.shape[2], feat_depth.shape[3]) 
        depth_logit = self.depth(x)

        preds = {
            'depth_logit': depth_logit,
            'feat_depth': feat_depth,
        }

        if self.seg_2d:
            # 2D-3D feature lifting
            img_feats = []
            for i in range(bs):
                img_feats.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
            img_feats = torch.cat(img_feats, 0)

            seg_logit = self.net_seg_2d(img_feats)

            preds['seg_logit'] = seg_logit

        return preds


def test_Net2DFeat():
    # 2D
    batch_size = 2
    img_width = 1220 // 2
    img_height = 370 // 2

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(
        batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(
        batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)
    print(img_indices.shape)
    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DFeat()

    net_2d.cuda()
    out_dict = net_2d(img, img_indices)
    for k, v in out_dict.items():
        print('Net2DFeat:', k, v.shape)


if __name__ == '__main__':
    test_Net2DFeat()
