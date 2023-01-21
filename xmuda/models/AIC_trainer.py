import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
from xmuda.models.AICNet import SSC_RGBD_AICNet
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.models.ssc_loss import CE_ssc_loss as compute_ssc_loss
import xmuda.data.semantic_kitti.io_data as SemanticKittiIO
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F


class AICTrainer(pl.LightningModule):
    def __init__(self,
                 n_classes,
                 output_scene_size,
                 full_scene_size,
                 class_names,
                 class_weights,
                 dataset, 
                 save_data_for_submission=False):
        super().__init__()
        self.save_data_for_submission = save_data_for_submission

        self.n_classes = n_classes
        self.class_names = class_names
        self.class_weights = class_weights
        self.output_scene_size = output_scene_size
        self.full_scene_size = full_scene_size
        self.aic_net = SSC_RGBD_AICNet(full_scene_size, output_scene_size, n_classes)
        self.dataset = dataset
        
        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)
        self.test_metrics_2 = SSCMetrics(self.n_classes)

        self.losses = []
        self.loss_sscs = []

    def forward(self, batch):
        rgb = batch['img'].float().cuda().contiguous()
        position = batch['mapping_1_1'].long().contiguous()
        depth = batch['depth'].float().contiguous()

        ssc_logit = self.aic_net(depth, rgb, position)
        out = {}
        out["ssc_logit"] = ssc_logit

        return out

    def step(self, batch, step_type, metric, metric_2=None):
        pred = self(batch)
        ssc_logit = pred['ssc_logit'].contiguous()
        bs = ssc_logit.shape[0]
        y_pred = ssc_logit.detach().cpu().numpy()
        if self.save_data_for_submission and self.dataset == "kitti" and step_type == "test":
            inv_map = SemanticKittiIO.get_inv_map()
            root_path = '/gpfsscratch/rech/kvd/uyl37fq/test_pred/AICNet/sequences/{}/predictions'
            for idx in range(y_pred.shape[0]):
                pred_label = np.argmax(y_pred[idx], axis=0).reshape(-1)
                pred_label = inv_map[pred_label].astype(np.uint16)
                frame_id = batch['frame_id'][idx]
                sequence = batch['sequence'][idx]
                folder = root_path.format(sequence)
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, "{}.label".format(frame_id))
                with open(save_path, 'wb') as f:
                    pred_label.tofile(f)
                    print("saved to", save_path)
        else:
            if self.output_scene_size[0] == self.full_scene_size[0]:
                target = batch['target'].contiguous()
            else:
                target = batch['ssc_label_1_4'].contiguous()

            loss = compute_ssc_loss(ssc_logit, target, self.class_weights.type_as(ssc_logit))

            y_true = target.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
#            frame_ids = batch['frame_id']
#            sequences = batch['sequence']
            metric.add_batch(y_pred, y_true)

            self.log(step_type + "/loss", loss.item(), on_epoch=True)

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
                self.log("{}_SemIoU/{}".format(prefix, class_name), stats['iou_ssc'][i])
            self.log("{}/mIoU".format(prefix), stats['iou_ssc_mean'])
            self.log("{}/IoU".format(prefix), stats['iou'])
            self.log("{}/Precision".format(prefix), stats['precision'])
            self.log("{}/Recall".format(prefix), stats['recall'])
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
#        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
#        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, last_epoch=-1)
#        return [optimizer], [scheduler]
        return optimizer
