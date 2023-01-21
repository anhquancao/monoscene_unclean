import pytorch_lightning as pl
import torch
import torch.nn as nn
import xmuda.data.semantic_kitti.io_data as SemanticKittiIO
from xmuda.models.LMSCNet import LMSCNet
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.models.ssc_loss import CE_ssc_loss as compute_ssc_loss
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F
import os
from time import time


class LMSCTrainer(pl.LightningModule):
    def __init__(self,
                 n_classes,
                 class_names,
                 full_scene_size,
                 output_scene_size,
                 dataset,
                 in_channels,
                 save_data_for_submission=False):
        super().__init__()
        self.save_data_for_submission = save_data_for_submission
        self.n_classes = n_classes
        self.dataset = dataset
        self.class_names = class_names
#        self.class_weights = class_weights
        if self.dataset == "kitti":
            epsilon_w = 0.001  # eps to avoid zero division
            self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05, 6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05, 2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07, 2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08, 2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
            class_weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))
        elif self.dataset == "NYU":
#            class_weights = torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            epsilon_w = 0.001  # eps to avoid zero division
            self.class_frequencies = np.array([43744234, 80205, 1070052, 905632, 116952, 180994, 436852, 279714, 254611, 28247, 1805949, 850724])
            class_weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))
        self.register_buffer('class_weights',  class_weights)

        self.net = LMSCNet(n_classes, output_scene_size, full_scene_size, in_channels)
        self.full_scene_size = full_scene_size
        self.output_scene_size = output_scene_size
        
        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)
        self.test_metrics_2 = SSCMetrics(self.n_classes)


    def forward(self, batch):
#        x = batch['tsdf_1_1'].float().cuda()
#        if self.full_scene_size[0] == self.output_scene_size[0]:
        x = batch['occ_1_1'].float().cuda()
#        else:
#            x = batch['occ_1_4'].float().cuda()

        if self.dataset == "NYU":
            x = x.permute(0, 2, 1, 3)
            ssc_logit, _ = self.net(x)
            ssc_logit = ssc_logit.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
        elif self.dataset == "kitti":
            x = x.permute(0, 3, 1, 2)
            ssc_logit, _ = self.net(x)
            ssc_logit = ssc_logit.permute(0, 1, 3, 4, 2)
        out = {}
        out["ssc_logit"] = ssc_logit
#        print(ssc_logit.shape)

        return out

    def step(self, batch, step_type, metric, metric_2=None):
        
        pred = self(batch)
        ssc_logit = pred['ssc_logit']
        bs = ssc_logit.shape[0]
        y_pred = ssc_logit.detach().cpu().numpy()
        
        # print(self.save_data_for_submission)
        # if self.save_data_for_submission and self.dataset == "kitti" and step_type == "test":
        #     inv_map = SemanticKittiIO.get_inv_map()
        #     root_path = '/gpfsscratch/rech/kvd/uyl37fq/test_pred/LMSCNet/sequences/{}/predictions'
        #     for idx in range(y_pred.shape[0]):
        #         pred_label = np.argmax(y_pred[idx], axis=0).reshape(-1)
        #         pred_label = inv_map[pred_label].astype(np.uint16)
        #         frame_id = batch['frame_id'][idx]
        #         sequence = batch['sequence'][idx]
        #         folder = root_path.format(sequence)
        #         os.makedirs(folder, exist_ok=True)
        #         save_path = os.path.join(folder, "{}.label".format(frame_id))
        #         with open(save_path, 'wb') as f:
        #             pred_label.tofile(f)
        #             print("saved to", save_path)
        #     print("test")
        # else:
        if self.output_scene_size[0] == self.full_scene_size[0]: 
            target = batch['target']
        else:
            target = batch['ssc_label_1_4']

        # start = time()
        loss = compute_ssc_loss(ssc_logit, target, self.class_weights.type_as(ssc_logit))
        # print(f'Time taken to run: {time() - start} seconds')

        y_true = target.cpu().numpy()
        _, y_pred = ssc_logit.max(dim=1)
        y_pred = y_pred.detach().cpu().numpy()
        # y_pred = np.argmax(y_pred, axis=1)

        metric.add_batch(y_pred, y_true)

        self.log(step_type + '/loss', loss.item(), on_epoch=True, sync_dist=True)
        
        if metric_2 is not None:
            if self.dataset == "kitti":
                fov = torch.stack(batch["valid_pix_1"]).reshape(bs,
                                                                self.output_scene_size[0],
                                                                self.output_scene_size[1],
                                                                self.output_scene_size[2])
                fov = fov.detach().cpu().numpy()
                y_pred[fov == 0] = 0     # 0 empty
                metric_2.add_batch(y_pred, y_true, fov)
            else:
                tsdf = batch['tsdf_1_4'].cpu().numpy()
                nonempty = (tsdf < 0.1) & (tsdf != 0)
                metric_2.add_batch(y_pred, y_true, nonempty)
    
        return loss


    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)


    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)


    def validation_epoch_end(self, outputs):
        metric_list = [('train', self.train_metrics), ('val', self.val_metrics)] 

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log("{}_SemIoU/{}".format(prefix, class_name), stats['iou_ssc'][i], sync_dist=True)
            self.log("{}/mIoU".format(prefix), stats['iou_ssc_mean'], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats['iou'], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats['precision'], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats['recall'], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics, self.test_metrics_2)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [('test', self.test_metrics), ('test_2', self.test_metrics_2)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print("{:.4f}, {:.4f}, {:.4f}".format(stats['precision'] * 100, stats['recall'] * 100, stats['iou'] * 100))
            print("{}, ".format(classes))
            print(' '.join(["{:.4f}, "] * len(classes)).format(*(stats['iou_ssc'] * 100).tolist()))
            print("mIoU", "{:.4f}".format(stats['iou_ssc_mean'] * 100))
            print(stats['iou_ssc_mean'])
            metric.reset()

    def configure_optimizers(self):
#        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        lambda1 = lambda epoch: (0.98) ** (epoch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        return [optimizer], [scheduler]
