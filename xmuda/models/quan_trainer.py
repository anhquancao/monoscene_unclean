import pytorch_lightning as pl
import torch
from xmuda.models.modules import Net2DFeat, Net3DFeat, FuseNet
# from xmuda.models.lmscnet_SS import LMSCNet_SS
# from xmuda.models.lmscnet_lite import LMSCNet_SS_lite
from xmuda.models.LMSCNet import LMSCNet
from xmuda.common.utils.metrics import Metrics
import pickle

import numpy as np
import time


class RecNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feat_2d = Net2DFeat()
        self.feat_3d = Net3DFeat(backbone_3d_kwargs={'in_channels': 1})
        self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                           6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                           2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                           2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                           2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
        self.class_num = 20
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

    def forward(self, batch):
        # torch.cuda.synchronize()
        # t = time.perf_counter()
        x = [t.cuda() for t in batch['x']]
        # coords = x[0]
        img = batch['img'].cuda()
        bs = img.shape[0]
        img_indices = batch['img_indices']        
        coords_2d = batch['coords_2d']
        coords_3d = batch['coords_3d']

        # occupancy = batch['voxel_occupancy']
        # device = img.device

        feat_2d = self.feat_2d(img, img_indices)['feats']  # n_points, dim
        feat_3d = self.feat_3d(x)['feats']  # n_points, dim

        fused_feat = self.fuse_2d_3d(
            feat_2d, feat_3d, coords_2d, coords_3d, bs)

        # fused_feat = torch.cat(fused_feats, dim=0)
        fused_feat = fused_feat.transpose(2, 3)

        # with open('images/fuse.pkl', 'wb') as handle:
        #     d = {
        #         "fuse": fused_feat.detach().cpu().numpy(),
        #         "occ": occupancy.detach().cpu().numpy()
        #     }
        #     pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print(fused_feat.shape)
        out = self.lmscnet(fused_feat)
        # torch.cuda.synchronize()
        # ExecTime = time.perf_counter() - t
        # print("exec time", ExecTime)
        return out

    def training_step(self, batch, batch_idx):

        pred = self(batch)

        # print(pred.shape, batch['ssc_label_1_1'].shape)
        # print(torch.max(batch['ssc_label_1_1']))

        target = batch['ssc_label_1_4']
        loss = self.lmscnet.compute_loss(pred, target)
        # print(torch.sum(target))
        self.train_metrics.add_batch(prediction=pred, target=target)

        self.log('train/loss', loss.item())
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

        # print(pred)
        target = batch['ssc_label_1_4']
        loss = self.lmscnet.compute_loss(pred, target)
        # loss = self.bce_logits_loss(logits, occ_labels)
        self.log('val/loss', loss.item())

        self.val_metrics.add_batch(prediction=pred, target=target)

        # pred_occ_labels = (torch.sigmoid(logits) > 0.5).float()
        # acc = (pred_occ_labels == occ_labels).float().mean()
        # self.log('train/acc', acc.item())

    def validation_epoch_end(self, outputs):
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
