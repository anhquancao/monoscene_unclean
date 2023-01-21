import pytorch_lightning as pl
import torch
from xmuda.models.modules import Net2DFeat, Net3DFeat, FuseNet, Unproject
# from xmuda.models.lmscnet_SS import LMSCNet_SS
# from xmuda.models.lmscnet_lite import LMSCNet_SS_lite
from xmuda.models.LMSCNet import LMSCNet
from xmuda.common.utils.metrics import Metrics
import pickle
import torch.nn.functional as F

import numpy as np
import time


class RecNetSeg(pl.LightningModule):
    def __init__(self, autoweighted_loss=False):
        super().__init__()
        self.autoweighted_loss = autoweighted_loss
        self.class_num = 20
        self.feat_2d = Net2DFeat(num_classes=self.class_num)
        self.unproject = Unproject()
        self.feat_3d = Net3DFeat(
            num_classes=self.class_num, backbone_3d_kwargs={'in_channels': 1})
        self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                           6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                           2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                           2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                           2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
        self.class_weights_3d = torch.tensor([2.27951038, 1.64677176, 3.53100987, 3.23751825, 2.71882114, 2.6341965,
                                              3.2803943,  3.58393049, 3.76410611, 1.03863938, 1.98235208, 1.19941985,
                                              2.42065937, 1.27775841, 1.43042397, 1.,         2.29782778, 1.41580027,
                                              2.55994229, 3.12169241]).cuda()
        self.class_weights_2d = torch.tensor([[2.451535,   1.61722757, 3.76430508, 3.30470327, 2.70467675, 2.66727774,
                                               3.35100329, 3.53497056, 3.51571106, 1.,         2.12430303, 1.30751046,
                                               2.56581265, 1.4156228,  1.49546134, 1.05270659, 2.30553103, 1.53045148,
                                               2.55582273, 3.07786581, ]]).cuda()
        self.seg_sigma_2d = torch.nn.Parameter(
            torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)
        self.seg_sigma_3d = torch.nn.Parameter(
            torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)
        self.sigma_ssc = torch.nn.Parameter(
            torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)

        self.fuse_2d_3d = FuseNet(
            in_channels_2d=64,
            in_channels_3d=16,
            hidden_dim=128)
        # self.lmscnet = LMSCNet_SS(
        #     class_num=self.class_num,
        #     class_frequencies=self.class_frequencies)
        self.lmscnet = LMSCNet(
            class_num=self.class_num,
            class_frequencies=self.class_frequencies,
            in_channels=32)

        self.train_metrics = Metrics(self.class_num)
        self.val_metrics = Metrics(self.class_num)

        self.losses = []
        self.loss_seg_2ds = []
        self.loss_seg_3ds = []
        self.loss_sscs = []
        # self.loss_depths = []

    def forward(self, batch):
        x = [t.cuda() for t in batch['x']]

        img = batch['img'].cuda()
        bs = img.shape[0]
        img_indices = batch['img_indices']
        coords_2d = batch['coords_2d']
        # coords_3d = batch['coords_3d']

        pred_2d = self.feat_2d(img, img_indices)
        # feat_2d = pred_2d['feats']  # n_points, dim
        seg_logit_2d = pred_2d['seg_logit']
        depth_logit = pred_2d['depth_logit']
        feat_depth = pred_2d['feat_depth']
        depth_prob = torch.softmax(depth_logit, dim=1)
        feat_depth = feat_depth * depth_prob

        K_inv = batch['K_inv'][0].cuda()
        T_inv = torch.from_numpy(batch['T_inv'][0]).cuda()
        pred_unproject = self.unproject(
            feat_depth, K_inv, T_inv)
        unprojected_feat = pred_unproject['unprojected_feat']
        # print(feat_depth.shape, depth_logit.shape)

        pred_3d = self.feat_3d(x)  # n_points, dim
        feat_3d = pred_3d['feats']
        seg_logit_3d = pred_3d['seg_logit']

        # fused_feat = self.fuse_2d_3d(
        #     feat_2d, feat_3d, coords_2d, coords_3d, bs)

        fused_feat = unprojected_feat.transpose(2, 3)

        ssc_logit = self.lmscnet(fused_feat)

        return {
            "seg_logit_3d": seg_logit_3d,
            "seg_logit_2d": seg_logit_2d,
            "ssc_logit": ssc_logit,
            "depth_logit": depth_logit
        }

    def training_step(self, batch, batch_idx):

        pred = self(batch)

        ssc_logit = pred['ssc_logit']
        seg_logit_3d = pred['seg_logit_3d']
        seg_logit_2d = pred['seg_logit_2d']
        # depth_logit = pred['depth_logit']
        target = batch['ssc_label_1_4']

        loss_ssc = self.lmscnet.compute_loss(ssc_logit, target)

        loss_seg_2d = F.cross_entropy(
            seg_logit_2d, batch['seg_label_2d'], self.class_weights_2d)
        loss_seg_3d = F.cross_entropy(
            seg_logit_3d, batch['seg_label_3d'], self.class_weights_3d)
        # loss_depth = F.cross_entropy(depth_logit, batch['depth_class'])

        if self.autoweighted_loss:
            factor_seg_2d = 1.0 / (self.seg_sigma_2d**2)
            factor_seg_3d = 1.0 / (self.seg_sigma_3d**2)
            factor_ssc = 1.0 / (self.sigma_ssc**2)
            loss = factor_seg_2d * loss_seg_2d + \
                factor_seg_3d * loss_seg_3d + \
                factor_ssc * loss_ssc +\
                2 * torch.log(self.seg_sigma_2d) + \
                2 * torch.log(self.seg_sigma_3d) +\
                2 * torch.log(self.sigma_ssc)
        else:
            loss = loss_seg_2d + loss_seg_3d + loss_ssc

        # loss = loss_ssc + loss_seg_3d + loss_seg_2d

        self.train_metrics.add_batch(prediction=ssc_logit, target=target)

        self.log('train/sigma_2d', self.seg_sigma_2d.item())
        self.log('train/sigma_3d', self.seg_sigma_3d.item())
        self.log('train/sigma_ssc', self.sigma_ssc.item())

        self.log('train/loss', loss.item())
        self.log('train/loss_ssc', loss_ssc.item())
        self.log('train/loss_seg_2d', loss_seg_2d.item())
        self.log('train/loss_seg_3d', loss_seg_3d.item())
        # self.log('train/loss_depth', loss_depth.item())
        self.log("train/mIoU", self.train_metrics.get_semantics_mIoU().item())
        self.log("train/IoU", self.train_metrics.get_occupancy_IoU().item())
        self.log("train/Precision",
                 self.train_metrics.get_occupancy_Precision().item())
        self.log("train/Recall", self.train_metrics.get_occupancy_Recall().item())
        self.log("train/F1", self.train_metrics.get_occupancy_F1().item())
        self.train_metrics.reset_evaluator()
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)

        ssc_logit = pred['ssc_logit']
        seg_logit_3d = pred['seg_logit_3d']
        seg_logit_2d = pred['seg_logit_2d']
        depth_logit = pred['depth_logit']
        target = batch['ssc_label_1_4']

        loss_ssc = self.lmscnet.compute_loss(ssc_logit, target)
        # print(seg_logit_2d.shape, batch['seg_label_2d'].shape)
        # print(batch['seg_label_2d'].max(0))
        loss_seg_2d = F.cross_entropy(seg_logit_2d, batch['seg_label_2d'])
        loss_seg_3d = F.cross_entropy(seg_logit_3d, batch['seg_label_3d'])
        # loss_depth = F.cross_entropy(depth_logit, batch['depth_class'])
        loss = loss_ssc + loss_seg_3d + loss_seg_2d

        self.losses.append(loss.item())
        # self.loss_depths.append(loss_depth.item())
        self.loss_seg_2ds.append(loss_seg_2d.item())
        self.loss_seg_3ds.append(loss_seg_3d.item())
        self.loss_sscs.append(loss_ssc.item())
        self.val_metrics.add_batch(prediction=ssc_logit, target=target)

    def validation_epoch_end(self, outputs):
        self.log("val/loss", np.mean(self.losses))
        self.log("val/loss_seg_2d", np.mean(self.loss_seg_2ds))
        self.log("val/loss_seg_3d", np.mean(self.loss_seg_3ds))
        self.log("val/loss_ssc", np.mean(self.loss_sscs))
        # self.log("val/loss_depth", np.mean(self.loss_depths))
        self.losses = []
        self.loss_seg_2ds = []
        self.loss_seg_3ds = []
        self.loss_sscs = []
        # self.loss_depths = []

        self.log("val/mIoU", self.val_metrics.get_semantics_mIoU().item())
        self.log("val/IoU", self.val_metrics.get_occupancy_IoU().item())
        self.log("val/Precision",
                 self.val_metrics.get_occupancy_Precision().item())
        self.log("val/Recall", self.val_metrics.get_occupancy_Recall().item())
        self.log("val/F1", self.val_metrics.get_occupancy_F1().item())
        self.val_metrics.reset_evaluator()

        # self.log("train/mIoU", self.train_metrics.get_semantics_mIoU().item())
        # self.log("train/IoU", self.train_metrics.get_occupancy_IoU().item())
        # self.log("train/Precision", self.train_metrics.get_occupancy_Precision().item())
        # self.log("train/Recall", self.train_metrics.get_occupancy_Recall().item())
        # self.log("train/F1", self.train_metrics.get_occupancy_F1().item())
        # self.train_metrics.reset_evaluator()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
