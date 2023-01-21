import pytorch_lightning as pl
import torch
from xmuda.models.encoder import Encoder
from xmuda.models.LMSCNet import LMSCNet
from xmuda.models.parallel_lmscnet import ParallelLMSCNet
from xmuda.common.utils.metrics import Metrics
from xmuda.models.ssc_loss import compute_ssc_loss
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F


class SSC2d(pl.LightningModule):
    def __init__(self,
                 num_depth_classes,
                 preprocess_dir,
                 edge_extractor=None,
                 edge_rgb_post_process=None,
                 seg_2d=False,
                 n_lmscnet_encoders=4,
                 branch="both",
                 shared_lmsc_encoder=False,
                 autoweighted_loss=False):
        super().__init__()
        assert branch in ['both', 'spatial', 'feature'], 'invalid option for branch'

        self.num_depth_classes = num_depth_classes
        self.seg_2d = seg_2d
        self.n_lmscnet_encoders = n_lmscnet_encoders
        self.branch = branch

#        with open(osp.join(preprocess_dir, "unproject_{}.pkl".format(num_depth_classes)), 'rb') as f:
#            self.mapping_2d_3d = pickle.load(f)
#
        with open(osp.join(preprocess_dir, "visible_voxels.pkl"), 'rb') as f:
            self.invisible_voxels = pickle.load(f)

        self.autoweighted_loss = autoweighted_loss
        self.class_num = 20

        self.encoder = Encoder(
            num_classes=self.class_num, 
            num_depth_classes=num_depth_classes,
            edge_extractor=edge_extractor,
            seg_2d=seg_2d,
            edge_rgb_post_process=edge_rgb_post_process,
            n_lmscnet_encoders=n_lmscnet_encoders
        )

        self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                           6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                           2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                           2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                           2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])

        self.class_weights_2d = torch.tensor([[2.451535,   1.61722757, 3.76430508, 3.30470327, 2.70467675, 2.66727774,
                                               3.35100329, 3.53497056, 3.51571106, 1.,         2.12430303, 1.30751046,
                                               2.56581265, 1.4156228,  1.49546134, 1.05270659, 2.30553103, 1.53045148,
                                               2.55582273, 3.07786581, ]]).cuda()
        self.seg_sigma_2d = torch.nn.Parameter(
            torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)
        self.sigma_ssc = torch.nn.Parameter(
            torch.Tensor(1).uniform_(0.2, 1), requires_grad=True)

        
        self.parallel_lmscnet = ParallelLMSCNet(
            class_num=self.class_num,
            class_frequencies=self.class_frequencies,
            n_encoders=n_lmscnet_encoders,
            shared_lmsc_encoder=shared_lmsc_encoder,
            in_channels=32)

#        self.lmscnet = LMSCNet(
#            class_num=self.class_num,
#            class_frequencies=self.class_frequencies,
#            in_channels=32)

        self.train_metrics = Metrics(self.class_num)
        self.val_metrics = Metrics(self.class_num)
        self.train_metrics_visible = Metrics(self.class_num)
        self.val_metrics_visible = Metrics(self.class_num)

        self.losses = []
        self.loss_seg_2ds = []
        self.loss_sscs = []

    def forward(self, batch):

        scenes = batch['scene']
        img = batch['img'].cuda()
        edge = batch['edge'].cuda()
        bs = img.shape[0]
        img_indices = batch['img_indices']
        edge_sparse_coord = batch['edge_sparse_coord']
        edge_sparse_feat = batch['edge_sparse_feat']

        pred_2d = self.encoder(img, img_indices, edge, edge_sparse_coord, edge_sparse_feat)

        if self.seg_2d:
            seg_logit_2d = pred_2d['seg_logit']

        depth_logit = pred_2d['depth_logit']
        feat_depth = pred_2d['feat_depth']

        depth_prob = torch.sigmoid(depth_logit)
        depth_prob = depth_prob.unsqueeze(1)

        if self.branch == "both":
            # [4, 4, 16, 185, 610]
            img_feat = feat_depth * depth_prob
        elif self.branch == "feature":
            img_feat = feat_depth
        elif self.branch == "spatial":
            img_feat = depth_prob

        img_feat = img_feat.transpose(3, 4) # (bs, n_lmsc_encoder, feat_dim, 610, 185)
        img_feat_flatten = img_feat.reshape(bs, img_feat.shape[1], img_feat.shape[2], -1) # (bs, 4, 16, 610 * 185)

        unprojected_feat = torch.zeros(
            (bs, self.n_lmscnet_encoders, 256 * 256 * 32), device=img.device)

        for i in range(bs):
            # project features 2d to 3d
            voxel_indices = batch['voxel_indices'][i] 
            pixel_indices_per_voxel = batch['pixel_indices_per_voxel'][i] # (n_voxels, n_pixels_per_voxel, 3) 
            n_voxels, n_pixels_per_voxel, _ = pixel_indices_per_voxel.shape
            
            # (4, 16, 610 * 185)
            img_feat_flatten_item = img_feat_flatten[i]

            # (n_voxels, 64)
            pixel_indices_per_voxel_flatten = pixel_indices_per_voxel[:, :, 0] * 185 + pixel_indices_per_voxel[:, :, 1]
            temp = pixel_indices_per_voxel_flatten.reshape(-1)

            # (4, 16, n_voxels, n_pixel_per_voxel)
            voxel_with_pixel_feats = img_feat_flatten_item[:, :, temp].reshape(img_feat_flatten_item.shape[0], 
                                                                               img_feat_flatten_item.shape[1],
                                                                               n_voxels, 
                                                                               n_pixels_per_voxel)

            # (n_voxels, 16, 4, n_pixel_per_voxel)
            voxel_with_pixel_feats = voxel_with_pixel_feats.permute(2, 1, 0, 3)

            voxel_feats, _ = voxel_with_pixel_feats.max(3) # (n_voxels, 16, n_lmsc_encoders)

            depth_idx = voxel_indices[:, 3].reshape(voxel_feats.shape[0], 1, 1).expand(-1, -1, voxel_feats.shape[2]) 

            # (n_voxels,1, 4)
            voxel_feats = torch.gather(voxel_feats, 1, depth_idx)

            # (n_voxels, 4)
            voxel_feats = voxel_feats.squeeze()

            voxel_indices_flatten = voxel_indices[:, 0] * (256 * 32) + voxel_indices[:, 1] * 32 + voxel_indices[:, 2]
            unprojected_feat[i, :, voxel_indices_flatten] = voxel_feats.T 
        
        unprojected_feat = unprojected_feat.reshape(bs, self.n_lmscnet_encoders, 256, 256, 32)

#        for i in range(bs):
#            scene = scenes[i]
#            voxel_indices = self.mapping_2d_3d[scene]['voxel_indices']
#            voxel_to_pixel_indices = self.mapping_2d_3d[scene]['voxel_to_pixel_indices']
#            img_indices = self.mapping_2d_3d[scene]['img_indices']
#            voxel_indices = torch.tensor(voxel_indices).long()
#            img_indices = torch.tensor(img_indices).long()
#            unprojected_feat[i, :,
#                             voxel_indices[:, 0],
#                             voxel_indices[:, 1],
#                             voxel_indices[:, 2]] += img_feat[i, :, img_indices[:, 2], img_indices[:, 1], img_indices[:, 0]]
#
        unprojected_feat = unprojected_feat.transpose(3, 4)
        unprojected_feats = [unprojected_feat[:, i, :, :, :] for i in range(unprojected_feat.shape[1])]

        ssc_logit = self.parallel_lmscnet(unprojected_feats)
        res = {
        #    "seg_logit_2d": seg_logit_2d,
            "ssc_logit": ssc_logit
        }
        if self.seg_2d:
            res['seg_logit_2d'] = seg_logit_2d

        return res

    def training_step(self, batch, batch_idx):

        pred = self(batch)

        ssc_logit = pred['ssc_logit']
        target = batch['ssc_label_1_4']

        loss_ssc = compute_ssc_loss(ssc_logit, target, self.class_frequencies)
        loss = loss_ssc

        if self.seg_2d:
            seg_logit_2d = pred['seg_logit_2d']
            loss_seg_2d = F.cross_entropy(seg_logit_2d, batch['seg_label_2d'], self.class_weights_2d)
            loss = loss_seg_2d + loss_ssc

        self.train_metrics.add_batch(prediction=ssc_logit, target=target)
        self.train_metrics_visible.add_batch(prediction=ssc_logit,
                                             target=target,
                                             scenes=batch['scene'],
                                             invisible_data_dict=self.invisible_voxels)

        self.log('train/sigma_2d', self.seg_sigma_2d.item())
        self.log('train/sigma_ssc', self.sigma_ssc.item())

        self.log('train/loss', loss.item())
        self.log('train/loss_ssc', loss_ssc.item())

        if self.seg_2d:
            self.log('train/loss_seg_2d', loss_seg_2d.item())

        for metrics, suffix in [(self.train_metrics, ""), (self.train_metrics_visible, "_visible")]:

            self.log("train/mIoU" + suffix,
                     metrics.get_semantics_mIoU().item())
            self.log("train/IoU" + suffix, metrics.get_occupancy_IoU().item())
            self.log("train/Precision" + suffix,
                     metrics.get_occupancy_Precision().item())
            self.log("train/Recall" + suffix,
                     metrics.get_occupancy_Recall().item())
            self.log("train/F1" + suffix, metrics.get_occupancy_F1().item())
            metrics.reset_evaluator()
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)

        ssc_logit = pred['ssc_logit']
        target = batch['ssc_label_1_4']

        loss_ssc = compute_ssc_loss(ssc_logit, target, self.class_frequencies)

        loss = loss_ssc
        if self.seg_2d:
            seg_logit_2d = pred['seg_logit_2d']
            loss_seg_2d = F.cross_entropy(seg_logit_2d, batch['seg_label_2d'], self.class_weights_2d)
            loss = loss_seg_2d + loss_ssc

        self.losses.append(loss.item())
        if self.seg_2d:
            self.loss_seg_2ds.append(loss_seg_2d.item())
        self.loss_sscs.append(loss_ssc.item())
        self.val_metrics.add_batch(prediction=ssc_logit, target=target)
        self.val_metrics_visible.add_batch(prediction=ssc_logit,
                                           target=target,
                                           scenes=batch['scene'],
                                           invisible_data_dict=self.invisible_voxels)

    def validation_epoch_end(self, outputs):
        self.log("val/loss", np.mean(self.losses))
        if self.seg_2d:
            self.log("val/loss_seg_2d", np.mean(self.loss_seg_2ds))
        self.log("val/loss_ssc", np.mean(self.loss_sscs))

        self.losses = []
        self.loss_seg_2ds = []
        self.loss_sscs = []

        for metrics, suffix in [(self.val_metrics, ""), (self.val_metrics_visible, "_visible")]:
            self.log("val/mIoU" + suffix,
                     metrics.get_semantics_mIoU().item())
            self.log("val/IoU" + suffix, metrics.get_occupancy_IoU().item())
            self.log("val/Precision" + suffix,
                     metrics.get_occupancy_Precision().item())
            self.log("val/Recall" + suffix,
                     metrics.get_occupancy_Recall().item())
            self.log("val/F1" + suffix, metrics.get_occupancy_F1().item())
            metrics.reset_evaluator()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
