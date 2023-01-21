import pytorch_lightning as pl
import torch
import torch.nn as nn

#from xmuda.models.decoder3D_aux_res import Decoder3D
from xmuda.models.decoder3D_v2 import Decoder3D
#from xmuda.models.decoder3D_1_8 import Decoder3D
from xmuda.models.resnet34_unet import UNetResNet34
from xmuda.common.utils.metrics import Metrics
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.models.sketch_init import init_weight, group_weight
from xmuda.models.ssc_loss import  focal_ssc_loss, MCA_ssc_loss, CE_ssc_loss, construct_ideal_affinity_matrix, AffinityLoss, ClassRelationLoss, multiscale_mca_loss, compute_class_proportion_loss
from xmuda.models.SSCNet_decoder import SSCNetDecoder
from xmuda.models.project2d3d_layer import Project2D3D
from xmuda.models.projectv2 import Project2D3Dv2
from xmuda.models.projectROI import Project2D3DROIAlign
#from xmuda.data.NYU.params import classes
from xmuda.models.CP_implitcit_loss import compute_loss_non_empty, compute_CP_implicit_loss, compute_pairwise_CP_implicit_loss, total_prob_constraint
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F
from xmuda.models.uefficientnet import UEfficientNetEncoder
from torch.optim.lr_scheduler import MultiStepLR
from collections import defaultdict

class SSC2dProj3d2d(pl.LightningModule):
    def __init__(self,
                 preprocess_dir,
                 n_classes,
                 class_names,
                 features,
                 class_weights,
                 class_relation_weights,
                 scene_sizes,
                 full_scene_size,
                 output_scene_size,
                 dataset,
                 max_k=256,
                 context_prior=None,
                 CP_res="1_16",
                 project_res=[],
                 CE_relation_loss=False,
                 MCA_relation_loss=False,
                 CE_ssc_loss=True,
                 MCA_ssc_loss=True,
                 net_2d_num_features=2048,
                 MCA_ssc_loss_type="one_minus",
                 rgb_encoder="UEffiNet",
                 lr=1e-4,
                 weight_decay=1e-4):
        super().__init__()
        assert rgb_encoder in ["UResNet", "UEffiNet"]
        assert context_prior in [None, "CP", "CRCP", "RP", "CPImplicit"]

        self.dataset = dataset
        self.class_names = class_names
        self.scene_sizes = scene_sizes
        self.full_scene_size = full_scene_size
        self.output_scene_size = output_scene_size
        self.CE_relation_loss = CE_relation_loss
        self.MCA_relation_loss = MCA_relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.MCA_ssc_loss = MCA_ssc_loss
        self.MCA_ssc_loss_type = MCA_ssc_loss_type
        self.class_relation_weights = class_relation_weights

        self.max_k = max_k

        self.class_weights = class_weights
        self.context_prior = context_prior
        self.CP_res = CP_res
        self.lr = lr
        self.weight_decay = weight_decay
#        self.scales = [4, 6, 8, 12, 16]
#        self.scales = [2, 4, 16]
        self.scales = [1, 2, 4, 8]
        projected_features = {
            "1": features[0],
            "2": features[0],
            "4": features[0],
            "8": features[0],
#            "16": features[0],
        }

        self.projects = {}
        self.convs = {}
        for scale in self.scales:
            key = str(scale)
            self.projects[key] = Project2D3D(
                tuple(i * 2 for i in self.scene_sizes[0]),
#                self.scene_sizes[0],
                projected_features[key],
                dataset=self.dataset)
            self.convs[key] = nn.Sequential(
#                nn.Conv3d(features[0], features[0], padding=0, kernel_size=1),
#                nn.BatchNorm3d(features[0]),
#                nn.ReLU(),
                nn.AvgPool3d(2, stride=2)
            )

        self.projects = nn.ModuleDict(self.projects)
        self.convs = nn.ModuleDict(self.convs)

        self.n_classes = n_classes
        if rgb_encoder == "UResNet":
            self.net_rgb = UNetResNet34(input_size=3)
            self.rgb_feat_channel = 64 
        elif rgb_encoder == "UEffiNet":
            self.net_rgb = UEfficientNetEncoder.build(out_feature=features[0], num_features=net_2d_num_features)
            self.rgb_feat_channel = 128 
        
        self.net_3d_decoder = Decoder3D(self.n_classes, 
                                        nn.BatchNorm3d,
                                        features=features,
                                        project_res=project_res,
                                        scene_sizes=self.scene_sizes,
                                        max_k=self.max_k,
                                        context_prior=context_prior)
        # log hyperparameters
        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.train_metrics_nonempty = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.val_metrics_nonempty = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)
        self.test_metrics_nonempty = SSCMetrics(self.n_classes)


    def forward(self, batch, max_k):

#        pts_cam_1_4s = batch['pts_cam_4']
#        pts_cam_1_8s = batch['pts_cam_8']
        pts_cam_1_16s = batch['pts_cam_16']
        target_1_16 = batch['ssc_label_1_16'] 
        mask_1_16s = (target_1_16 != 255).reshape(target_1_16.shape[0], -1)

        img = batch['img']
        bs = len(img)

        out = {}

        x_rgb = self.net_rgb(img)

        x3ds = defaultdict(list)
        if self.full_scene_size[0] == self.output_scene_size[0]:
            output_res = 1
        else:
#            output_res = 4
            output_res = 2
        for i in range(bs):
            for scale in self.scales:
                pix = batch["pix_{}".format(output_res)][i].to(self.device)#cuda()
                valid_pix = batch["valid_pix_{}".format(output_res)][i].to(self.device)#.cuda()
                pts_cam = batch["pts_cam_{}".format(output_res)][i].to(self.device)#.cuda()
                x3d = self.projects[str(scale)](x_rgb["1_" + str(scale)][i], pix // scale, valid_pix, pts_cam)
                x3ds[str(scale)].append(x3d) 

        input_dict = {
            "pts_cam_1_16": pts_cam_1_16s,
#            "pts_cam_1_4": pts_cam_1_4s,
            "max_k": max_k
        }

        for key in x3ds:
            input_dict["x3d_1_" + key] = self.convs[key](torch.stack(x3ds[key]))
#            input_dict["x3d_1_" + key] = torch.stack(x3ds[key])

        if self.context_prior == 'CRCP':
            input_dict['masks_1_16'] = mask_1_16s 


        out = self.net_3d_decoder(input_dict)
        
        return out 

    def step(self, batch, step_type, metric, metric_nonempty, max_k):
        bs = len(batch['img'])
        loss = 0
        out_dict = self(batch, max_k)
        ssc_logit = out_dict['ssc']
        if self.output_scene_size[0] == self.full_scene_size[0]:
            target = batch['ssc_label_1_1']
        else:
            target = batch['ssc_label_1_4']
        target_1_16 = batch['ssc_label_1_16']
#        targets = {
#            '1_4': batch['ssc_label_1_4'],
#            '1_8': batch['ssc_label_1_8'],
#            '1_16': batch['ssc_label_1_16'],
#        }

        if self.context_prior == 'CRCP':
            P_logits = out_dict['P_logits']
            list_topk_indices = out_dict['topk_indices']
            relate_probs = out_dict['relate_probs']
#            if self.logger is not None and self.current_epoch % 10 == 0:
#                tensorboard = self.logger.experiment
#                names = batch['scene']
##                for i in range(len(names)):
#                i = 0
#                tensorboard.add_histogram(step_type + str(self.current_epoch) + "/relate_prob_" + names[i], relate_probs[i])
#
            loss_relation = 0
            for i in range(bs):
#                mask = masks[i] 
#                A = As[i]
                P_logit = P_logits[i]
                topk_indices = list_topk_indices[i]
#                target_1_16_i = target_1_16s[i]
                loss_relation += compute_pairwise_CP_implicit_loss(P_logit, target_1_16[i], 
                                                                   topk_indices, self.class_relation_weights, 
                                                                   self.CE_relation_loss, self.MCA_relation_loss)
            loss_relation /= bs
            loss += loss_relation
            self.log(step_type + '/loss_relation', loss_relation.detach(), on_epoch=True, sync_dist=True)

#        target = targets['1_4']
#        target_1_16 = targets['1_16']
        class_weight = self.class_weights.type_as(ssc_logit)

        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_logit, target, class_weight)
            loss += loss_ssc
            self.log(step_type + '/loss_ssc', loss_ssc.detach(), on_epoch=True, sync_dist=True)
        
        if self.MCA_ssc_loss:
            loss_completion = MCA_ssc_loss(ssc_logit, target, self.MCA_ssc_loss_type)
            loss += loss_completion
            self.log(step_type + '/loss_completion', loss_completion.detach(), on_epoch=True, sync_dist=True)

#        loss_class_proportion = compute_class_proportion_loss(ssc_logit, target)

#            loss_multiscale_mca_8 = multiscale_mca_loss(ssc_logit, 
#                                                        targets['1_4'], targets["1_8"], 2)
#            loss += loss_multiscale_mca_8
#            self.log(step_type + '/loss_multiscale_mca_8', loss_multiscale_mca_8.detach(), on_epoch=True, sync_dist=True)
#            loss_multiscale_mca_16 = multiscale_mca_loss(ssc_logit, 
#                                                         targets['1_4'], targets["1_16"], 4)
#            self.log(step_type + '/loss_multiscale_mca_16', loss_multiscale_mca_16.detach(), on_epoch=True, sync_dist=True)
#            loss += loss_multiscale_mca_16

        y_true = target.cpu().numpy()
        y_pred = ssc_logit.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

#        if 'nonempty' in batch:
#            nonempty = batch['nonempty'].cpu().numpy()
#            y_pred[nonempty == 0] = 0     # 0 empty
#            metric_nonempty.add_batch(y_pred, y_true, nonempty)

        self.log(step_type + '/loss', loss.detach(), on_epoch=True, sync_dist=True)

        return loss
        
    def training_step(self, batch, batch_idx):
#        self.log('step', self.global_step, sync_dist=True)
        return self.step(batch, "train", self.train_metrics, self.train_metrics_nonempty, max_k=self.max_k) 

#    def training_step_end(self, batch_parts):
#        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics, self.val_metrics_nonempty, self.max_k) 
        

    def validation_epoch_end(self, outputs):
        metric_list = [('train', self.train_metrics), ('val', self.val_metrics)] 
        metric_list += [('train_nonempty', self.train_metrics_nonempty), ('val_nonempty', self.val_metrics_nonempty)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
#            tr(self.current_epoch) + "/" + prefix, stats['iou_ssc'])
            for i, class_name in enumerate(self.class_names):
                self.log("{}_SemIoU/{}".format(prefix, class_name), stats['iou_ssc'][i], sync_dist=True)
            self.log("{}/mIoU".format(prefix), stats['iou_ssc_mean'], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats['iou'], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats['precision'], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats['recall'], sync_dist=True)
        
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics, self.test_metrics_nonempty, self.max_k) 
        

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [('test', self.test_metrics), ('test_nonempty', self.test_metrics_nonempty)] 
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print("{:.4f}, {:.4f}, {:.4f}".format(stats['precision'] * 100, stats['recall'] * 100, stats['iou'] * 100))
            print("{}, ".format(classes))
            print(' '.join(["{:.4f}, "] * len(classes)).format(*(stats['iou_ssc'] * 100).tolist()))
            print("mIoU", "{:.4f}".format(stats['iou_ssc_mean'] * 100))
            print(stats['iou_ssc_mean'])
            metric.reset()

#        for prefix, metric in metric_list:
#            stats = metric.get_stats()
##            tr(self.current_epoch) + "/" + prefix, stats['iou_ssc'])
#            for i, class_name in enumerate(classes):
#                self.log("{}_SemIoU/{}".format(prefix, class_name), stats['iou_ssc'][i], sync_dist=True)
#            self.log("{}/mIoU".format(prefix), stats['iou_ssc_mean'], sync_dist=True)
#            self.log("{}/IoU".format(prefix), stats['iou'], sync_dist=True)
#            self.log("{}/Precision".format(prefix), stats['precision'], sync_dist=True)
#            self.log("{}/Recall".format(prefix), stats['recall'], sync_dist=True)
#            metric.reset()


    def configure_optimizers(self):
#        params = [{"params": self.net_rgb.encoder.parameters(), "lr": self.lr_2D_encoder},
#                  {"params": list(self.net_rgb.decoder.parameters()) + list(self.net_3d_decoder.parameters()), "lr": self.lr}]
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
#        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
#                                                      lr_lambda=lambda epoch: self.lr_power ** epoch)
#        scheduler = MultiStepLR(optimizer, milestones=[70], gamma=0.1)
#        return [optimizer], [scheduler]

