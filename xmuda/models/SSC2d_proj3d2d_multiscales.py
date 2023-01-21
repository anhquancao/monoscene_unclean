import pytorch_lightning as pl
import torch
import torch.nn as nn
from xmuda.models.network_3dsketch import Decoder3D
from xmuda.models.resnet34_unet import UNetResNet34
from xmuda.common.utils.metrics import Metrics
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.models.ssc_loss import compute_ssc_loss
from xmuda.models.SSCNet_decoder import SSCNetDecoder
from xmuda.models.project2d3d_layer import Project2D3D
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F
from xmuda.models.uefficientnet import UEfficientNetEncoder


class SSC2dProj3d2d(pl.LightningModule):
    def __init__(self,
                 preprocess_dir,
                 input_scene_size,
                 output_scene_size,
                 n_classes,
                 class_weights,
                 rgb_encoder="UResNet",
                 lr=1e-4,
                 same_lr=False,
                 seg_2d=True):
        super().__init__()
        assert rgb_encoder in ["UResNet", "UEffiNet"]

        self.seg_2d = seg_2d
        self.input_scene_size = input_scene_size
        self.output_scene_size = output_scene_size
        self.class_weights = class_weights
#        self.register_buffer("class_weights", class_weights)
        self.same_lr = same_lr
        self.lr = lr
        self.project2d3d_1_4 = Project2D3D((60, 36, 60))
        self.project2d3d_1_8 = Project2D3D((30, 18, 30))
        self.project2d3d_1_16 = Project2D3D((15, 9, 15))

#        with open(osp.join(preprocess_dir, "voxel_to_pixel_{}.pkl".format(0.8)), 'rb') as f:
#            self.map_3d2d = pickle.load(f)

#        with open(osp.join(preprocess_dir, "visible_voxels.pkl"), 'rb') as f:
#            self.invisible_voxels = pickle.load(f)

        self.n_classes = n_classes
        if rgb_encoder == "UResNet":
            self.net_rgb = UNetResNet34(input_size=3)
            self.rgb_feat_channel = 64 
        elif rgb_encoder == "UEffiNet":
            self.net_rgb = UEfficientNetEncoder.build()
            self.rgb_feat_channel = 128 
        
        self.in_channels_3d = self.rgb_feat_channel
#        self.net_3d_decoder = SSCNetDecoder(self.n_classes, in_channels=self.in_channels_3d)
        self.net_3d_decoder = Decoder3D(self.n_classes,
                                        nn.BatchNorm3d,
                                        feature=128,
                                        in_channels={
                                            '1_16': 256,
                                            '1_8': 128,
                                            '1_4': 128
                                        })

        self.train_metrics = {
            '1_4': SSCMetrics(self.n_classes),
            '1_8': SSCMetrics(self.n_classes),
            '1_16': SSCMetrics(self.n_classes),
        }
        self.val_metrics = {
            '1_4': SSCMetrics(self.n_classes),
            '1_8': SSCMetrics(self.n_classes),
            '1_16': SSCMetrics(self.n_classes),
        }

        self.train_metrics_nonempty = {
            '1_4': SSCMetrics(self.n_classes),
            '1_8': SSCMetrics(self.n_classes),
            '1_16': SSCMetrics(self.n_classes),
        }
        self.val_metrics_nonempty = {
            '1_4': SSCMetrics(self.n_classes),
            '1_8': SSCMetrics(self.n_classes),
            '1_16': SSCMetrics(self.n_classes),
        }

        self.val_losses = []
        self.loss_sscs = {
            "1_4": [],
            "1_8": [],
            "1_16": []
        }

#        g_x = torch.tensor(np.arange(0, self.input_scene_size[0]))
#        g_y = torch.tensor(np.arange(0, self.input_scene_size[1]))
#        g_z = torch.tensor(np.arange(0, self.input_scene_size[2]))
#        coords = torch.stack(torch.meshgrid(g_x, g_y, g_z))
#        self.coords = coords.cuda()

    def get_1x_lr_params(self):
        return self.net_rgb.parameters()

    def get_10x_lr_params(self):
        modules = [self.net_3d_decoder.get_parameters()]
        for m in modules:
            yield from m.parameters()

    def forward(self, batch):

        voxel_indices_1_4s = batch['voxel_indices_1_4']
        img_indices_1_4s = batch['img_indices_1_4']
        voxel_indices_1_8s = batch['voxel_indices_1_8']
        img_indices_1_8s = batch['img_indices_1_8']
        voxel_indices_1_16s = batch['voxel_indices_1_16']
        img_indices_1_16s = batch['img_indices_1_16']
        img = batch['img'].cuda()
        bs = img.shape[0]

        out = {}

        x_rgbs = self.net_rgb(img)

        x3d_1_4s = []
        x3d_1_8s = []
        x3d_1_16s = []
        for i in range(bs):
            voxel_indices_1_4 = voxel_indices_1_4s[i].cuda()
            img_indices_1_4 = img_indices_1_4s[i].cuda()
            voxel_indices_1_8 = voxel_indices_1_8s[i].cuda()
            img_indices_1_8 = img_indices_1_8s[i].cuda()
            voxel_indices_1_16 = voxel_indices_1_16s[i].cuda()
            img_indices_1_16 = img_indices_1_16s[i].cuda()

#            print("-1", x_rgbs[-1][i].shape)
#            print("-2", x_rgbs[-2][i].shape)
#            print("-3", x_rgbs[-3][i].shape)
            x3d_1_4 = self.project2d3d_1_4(x_rgbs[-1][i], voxel_indices_1_4, img_indices_1_4)
            x3d_1_8 = self.project2d3d_1_8(x_rgbs[-2][i], voxel_indices_1_8, img_indices_1_8 // 2)
            x3d_1_16 = self.project2d3d_1_16(x_rgbs[-3][i], voxel_indices_1_16, img_indices_1_16 // 4)

            x3d_1_4s.append(x3d_1_4)
            x3d_1_8s.append(x3d_1_8)
            x3d_1_16s.append(x3d_1_16)

        x3d_1_4 = torch.stack(x3d_1_4s)
        x3d_1_8 = torch.stack(x3d_1_8s)
        x3d_1_16 = torch.stack(x3d_1_16s)
        
        input_dict = {
            "x3d_1_4": x3d_1_4,
            "x3d_1_8": x3d_1_8,
            "x3d_1_16": x3d_1_16
        }

        out = self.net_3d_decoder(input_dict)
        
        return out

    def training_step(self, batch, batch_idx):

        pred = self(batch)

        ssc_logits = {
            '1_4': pred['ssc_logit_1_4'],
            '1_8': pred['ssc_logit_1_8'],
            '1_16': pred['ssc_logit_1_16']
        }
        targets = {
            '1_4': batch['ssc_label_1_4'],
            '1_8': batch['ssc_label_1_8'],
            '1_16': batch['ssc_label_1_16'],
        }
        loss = 0
        for key in ssc_logits:
            ssc_logit = ssc_logits[key]
            target = targets[key]
            class_weight = self.class_weights[key]
            loss_ssc= compute_ssc_loss(ssc_logit, target, class_weight)
            self.loss_sscs[key].append(loss_ssc.item())
            loss += loss_ssc


            y_true = target.cpu().numpy()
            y_pred = ssc_logit.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            self.train_metrics[key].add_batch(y_pred, y_true)

            if key == '1_4' and 'nonempty' in batch:
                nonempty = batch['nonempty'].cpu().numpy()
                y_pred[nonempty == 0] = 0     # 0 empty
                self.train_metrics_nonempty[key].add_batch(y_pred, y_true, nonempty)

        self.log('step', self.global_step)
        self.log('train/loss', loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)

        ssc_logits = {
            '1_4': pred['ssc_logit_1_4'],
            '1_8': pred['ssc_logit_1_8'],
            '1_16': pred['ssc_logit_1_16']
        }
        targets = {
            '1_4': batch['ssc_label_1_4'],
            '1_8': batch['ssc_label_1_8'],
            '1_16': batch['ssc_label_1_16'],
        }
        loss = 0
        for key in ssc_logits:
            ssc_logit = ssc_logits[key]
            target = targets[key]
            class_weight = self.class_weights[key]
            loss_ssc= compute_ssc_loss(ssc_logit, target, class_weight)
            self.loss_sscs[key].append(loss_ssc.item())
            loss += loss_ssc

            y_true = target.cpu().numpy()
            y_pred = ssc_logit.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            self.val_metrics[key].add_batch(y_pred, y_true)

            if key == '1_4' and 'nonempty' in batch:
                nonempty = batch['nonempty'].cpu().numpy()
                y_pred[nonempty == 0] = 0     # 0 empty
                self.val_metrics_nonempty[key].add_batch(y_pred, y_true, nonempty)
        self.val_losses.append(loss.item())


    def validation_epoch_end(self, outputs):
        self.log("val/loss", np.mean(self.val_losses))

        self.val_losses = []
        for key in self.loss_sscs:
            self.log("val/loss_" + key, np.mean(self.loss_sscs[key]))
            self.loss_sscs[key] = []

            val_metrics = self.val_metrics[key]
            val_metrics_nonempty = self.val_metrics[key]
            train_metrics = self.train_metrics[key]
            train_metrics_nonempty = self.train_metrics_nonempty[key]

            metric_list = [('train', train_metrics), ('val', val_metrics)] 
            if key == '1_4':
                metric_list += [('train_nonempty', train_metrics_nonempty), ('val_nonempty', val_metrics_nonempty)]

            for prefix, metric in metric_list:
                stats = metric.get_stats()
                if key != '1_4':
                   prefix += "_{}".format(key) 
                self.log("{}/mIoU".format(prefix), stats['iou_ssc_mean'])
                self.log("{}/IoU".format(prefix), stats['iou'])
                self.log("{}/Precision".format(prefix), stats['precision'])
                self.log("{}/Recall".format(prefix), stats['recall'])
            
            train_metrics.reset()
            train_metrics_nonempty.reset()
            val_metrics.reset()
            val_metrics_nonempty.reset()

    def configure_optimizers(self):
        params = self.parameters()

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
#        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
