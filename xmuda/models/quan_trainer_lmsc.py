import pytorch_lightning as pl
import torch
from xmuda.models.modules import Net2DFeat, Net3DFeat, FuseNet
from xmuda.models.LMSCNet import LMSCNet
from xmuda.common.utils.metrics import Metrics
import pickle
import numpy as np
import time
import os.path as osp


class RecNetLMSC(pl.LightningModule):
    def __init__(self, preprocess_dir):
        super().__init__()
        self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                           6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                           2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                           2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                           2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
        self.class_num = 20
        self.lmscnet = LMSCNet(
            class_num=self.class_num,
            class_frequencies=self.class_frequencies)

        with open(osp.join(preprocess_dir, "visible_voxels.pkl"), 'rb') as f:
            self.invisible_voxels = pickle.load(f)

        self.train_metrics = Metrics(self.class_num)
        self.val_metrics = Metrics(self.class_num)
        self.train_metrics_visible = Metrics(self.class_num)
        self.val_metrics_visible = Metrics(self.class_num)
        # tensorboard = self.logger.experiment

    def forward(self, batch):
        occupancy = batch['voxel_occupancy'].cuda()
        # n_points_3d = batch['n_points_3d']
#        img = batch['img']
#        bs = img.shape[0]
#        coords_3d = batch['coords_3d']
#        occupancy = torch.zeros(bs, 256, 256, 32, device=self.device)
#        # print(coords_3d.shape)
#        # prev = 0
#        for i in range(bs):
#            idx = coords_3d[:, 3] == i
#            b_coords = coords_3d[idx]
#            occupancy[i, b_coords[:, 0], b_coords[:, 1], b_coords[:, 2]] = 1.0
#            # prev = n_point
#        occupancy = occupancy.transpose(2, 3)

        out = self.lmscnet(occupancy)
        return out

    def training_step(self, batch, batch_idx):

        pred = self(batch)

        target = batch['ssc_label_1_4'].cuda()
        loss = self.lmscnet.compute_loss(pred, target)

        self.train_metrics.add_batch(prediction=pred, target=target)
        self.train_metrics_visible.add_batch(prediction=pred,
                                             target=target,
                                             scenes=batch['scene'],
                                             invisible_data_dict=self.invisible_voxels)

        self.log('train/loss', loss.item())
        self.log('train/loss_ssc', loss.item())
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

        target = batch['ssc_label_1_4'].cuda()
        loss = self.lmscnet.compute_loss(pred, target)

        # loss = self.bce_logits_loss(logits, occ_labels)
        self.log('val/loss', loss.item())
        self.log('val/loss_ssc', loss.item())

        self.val_metrics.add_batch(prediction=pred, target=target)
        self.val_metrics_visible.add_batch(prediction=pred,
                                           target=target,
                                           scenes=batch['scene'],
                                           invisible_data_dict=self.invisible_voxels)

        # pred_occ_labels = (torch.sigmoid(logits) > 0.5).float()
        # acc = (pred_occ_labels == occ_labels).float().mean()
        # self.log('train/acc', acc.item())

    def validation_epoch_end(self, outputs):
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
