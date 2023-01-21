import pytorch_lightning as pl
import torch
import os
import torch.nn as nn
import xmuda.data.semantic_kitti.io_data as SemanticKittiIO
#from xmuda.models.decoder3D_kitti_CP3DAt1_8 import Decoder3DKitti
from xmuda.models.decoder3D_kitti_v2 import Decoder3DKitti
from xmuda.models.decoder3D_simple_v2 import Decoder3D
from xmuda.models.resnet34_unet import UNetResNet34
from xmuda.common.utils.metrics import Metrics
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.models.sketch_init import init_weight, group_weight
from xmuda.models.ssc_loss import  focal_ssc_loss, MCA_ssc_loss, CE_ssc_loss, construct_ideal_affinity_matrix, AffinityLoss, ClassRelationLoss, multiscale_mca_loss, compute_class_proportion_loss, compute_class_proportion_from_2d, compute_class_proportion_klmax_loss, JSD_v2, KL, KL_sep, IoU_loss, JSD_smooth, JSD_nonzeros, JSD_sep
from xmuda.models.SSCNet_decoder import SSCNetDecoder
from xmuda.models.project2d3d_layer import Project2D3D, ProjectROIPool
from xmuda.models.CP_implitcit_loss import compute_mega_CP_loss, compute_super_CP_multilabel_loss, compute_CP_sliced_loss, compute_voxel_pairwise_rel_loss 
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F
from xmuda.models.uefficientnet import UEfficientNetEncoder
#from xmuda.models.uefficientnetv2 import UEfficientNetV2Encoder
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from collections import defaultdict
from xmuda.common.utils.js3cnet_metric import iouEval


class SSC2dLMSCNetProj(pl.LightningModule):
    def __init__(self,
                 preprocess_dir,
                 n_classes,
                 class_names,
                 features,
                 class_weights,
                 class_proportion_loss,
                 scene_sizes,
                 project_scale,
                 full_scene_size,
                 output_scene_size,
                 dataset,
                 corenet_proj=None,
                 lovasz=False,
                 optimize_iou=False,
                 n_relations=4,
                 context_prior=None,
                 CP_res="1_16",
                 project_res=[],
                 frustum_size=4,
                 CE_relation_loss=False,
                 MCA_relation_loss=False,
                 CE_ssc_loss=True,
                 MCA_ssc_loss=True,
                 MCA_ssc_loss_type="one_minus",
                 rgb_encoder="UEffiNet",
                 save_data_for_submission=False,
                 lr=1e-4,
                 weight_decay=1e-4):
        super().__init__()
        assert rgb_encoder in ["UResNet", "UEffiNet"]
        assert context_prior in [None, "CP", "CRCP", "RP", "CPImplicit"]

        self.lovasz = lovasz
        self.optimize_iou = optimize_iou
        self.project_res = project_res
        self.class_proportion_loss = class_proportion_loss
        self.relation_names = ["non_non_same", "non_non_diff", "empty_empty", "empty_non"]
        self.dataset = dataset
        self.frustum_size = frustum_size
        self.class_names = class_names
        self.scene_sizes = scene_sizes
        self.save_data_for_submission = save_data_for_submission
        self.full_scene_size = full_scene_size
        self.output_scene_size = output_scene_size
        self.CE_relation_loss = CE_relation_loss
        self.MCA_relation_loss = MCA_relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.MCA_ssc_loss = MCA_ssc_loss
        self.MCA_ssc_loss_type = MCA_ssc_loss_type
        self.project_scale = project_scale
        if self.dataset == "kitti":
            self.output_scale = int(self.project_scale / 2)
        elif self.dataset == "NYU":
            self.output_scale = 4

        self.class_weights = class_weights
        self.context_prior = context_prior
        self.CP_res = CP_res
        self.lr = lr
        self.weight_decay = weight_decay

        self.projects = {}
        self.scales = [1, 2, 4, 8]
        for scale in self.scales:
            key = str(scale)
            self.projects[key] = Project2D3D(
                self.scene_sizes[0],
                project_scale=self.project_scale,
                dataset=self.dataset)
        self.projects = nn.ModuleDict(self.projects)

        self.n_classes = n_classes
        if self.dataset == "NYU":
            feat_2D = features[0]
            self.net_3d_decoder = Decoder3D(self.n_classes,
                                            nn.BatchNorm3d,
                                            n_relations=n_relations,
                                            features=features,
                                            corenet_proj=None,
                                            scene_sizes=self.scene_sizes,
                                            context_prior=context_prior)
        elif self.dataset == "kitti":
            feat_2D = features[0] * 2
            self.net_3d_decoder = Decoder3DKitti(self.n_classes,
                                                 nn.BatchNorm3d,
                                                 project_scale=project_scale,
                                                 feature=features[0],
                                                 full_scene_size=self.full_scene_size,
                                                 context_prior=context_prior)
        if rgb_encoder == "UResNet":
            self.net_rgb = UNetResNet34(input_size=3)
        elif rgb_encoder == "UEffiNet":
            self.net_rgb = UEfficientNetEncoder.build(out_feature=feat_2D,
                                                      use_decoder=True)

        # log hyperparameters
        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.train_metrics_fov = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.val_metrics_fov = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)
        self.test_metrics_fov = SSCMetrics(self.n_classes)

        if self.dataset == "kitti":
            self.unroll = nn.Conv3d(1, features[0]*2, kernel_size=1, padding=0)
        else:
            self.unroll = nn.Conv3d(1, features[0], kernel_size=1, padding=0)

#        self.unroll2d = nn.Sequential(
#            nn.Conv2d(200, 3000, kernel_size=1, padding=0),
#            nn.BatchNorm2d(3000),
#            nn.ReLU()
#        )
#        self.unroll3d = nn.Sequential(
#            nn.Conv3d(50, 200, kernel_size=1, padding=0),
#            nn.BatchNorm3d(200),
#            nn.ReLU()
#        )


    def forward(self, batch):


        img = batch['img']
        bs = len(img)

        out = {}

        x_rgb = self.net_rgb(img)

        x2d = x_rgb["1_1"][:, None, :, :, :]
        if self.dataset == "NYU":
            x3d = F.interpolate(x2d, (60, 36, 60), mode="trilinear")
        else:
            x3d = F.interpolate(x2d, (128, 16, 128), mode="trilinear")
        x3d = x3d.permute(0, 1, 2, 4, 3)
        x3d = self.unroll(x3d)
#        print(x3d.shape)
#        x2d = x_rgb["1_1"]
#        x2d = F.interpolate(x2d, (36, 60), mode="bilinear")
#        x2d = self.unroll2d(x2d)
#        x3d = x2d.reshape(bs, 50, 60, 36, 60).permute(0, 1, 4, 3, 2)
#        x3d = self.unroll3d(x3d)

        input_dict = {
            "x3d": x3d
        }

        out = self.net_3d_decoder(input_dict)

        out['ssc_logit'] = out['ssc']

        return out


    def step(self, batch, step_type, metric, metric_fov):
        bs = len(batch['img'])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict['ssc']
        if step_type == "test" and self.save_data_for_submission:
            y_pred = ssc_pred.detach().cpu().numpy()
            inv_map = SemanticKittiIO.get_inv_map()
            root_path = '/gpfsscratch/rech/kvd/uyl37fq/test_pred/Our/sequences/{}/predictions'
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
            return

        if self.dataset == "kitti": 
            target = batch['target']
#            if self.project_scale == 2:
#                target = batch['ssc_label_1_1']
#            elif self.project_scale == 4:
#                target = batch['ssc_label_1_2']
        elif self.dataset == "NYU":
            target = batch['ssc_label_1_4']


        if self.context_prior == 'CRCP':
            P_logits = out_dict['P_logits']
            CP_mega_matrices = batch['CP_mega_matrices']
#            target_1_16 = batch['ssc_label_1_16']

            if self.CE_relation_loss:
#                print("==========CE_relation")
    #            loss_rel_ce, loss_rel_global = compute_super_CP_multilabel_loss(P_logits, CP_mega_matrices)
                loss_rel_ce = compute_super_CP_multilabel_loss(P_logits, CP_mega_matrices)
                loss += loss_rel_ce
    #            loss += loss_rel_ce + loss_rel_global
                self.log(step_type + '/loss_relation_ce_super', loss_rel_ce.detach(), on_epoch=True, sync_dist=True)
    #            self.log(step_type + '/loss_relation_global_super', loss_rel_global.detach(), on_epoch=True, sync_dist=True)



        class_weight = self.class_weights.type_as(batch['img'])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(step_type + '/loss_ssc', loss_ssc.detach(), on_epoch=True, sync_dist=True)
        
        if self.MCA_ssc_loss:
            loss_completion = MCA_ssc_loss(ssc_pred, target, self.MCA_ssc_loss_type)
            loss += loss_completion
            self.log(step_type + '/loss_completion', loss_completion.detach(), on_epoch=True, sync_dist=True)


        if self.optimize_iou:
            loss_iou = IoU_loss(ssc_pred, target, is_lovasz=self.lovasz)
            loss += loss_iou
            self.log(step_type + '/loss_iou', loss_iou.detach(), on_epoch=True, sync_dist=True)


        if self.class_proportion_loss and step_type != "test":
            if self.dataset == "NYU":
                n_frustums = self.frustum_size ** 2 
                local_frustums = torch.stack(batch["local_frustums_{}".format(self.output_scale)])
                local_frustums_cnt = torch.stack(batch["local_frustums_cnt_{}".format(self.output_scale)]).float()  # (2, n_frustums, 12)
#                print(local_frustums.shape)
#                local_frustums = torch.stack(batch["local_frustums_{}".format(self.output_scale)]).reshape(bs, n_frustums, 
#                                                                                                    self.output_scene_size[0], 
#                                                                                                    self.output_scene_size[2], 
#                                                                                                    self.output_scene_size[1]).permute(0, 1, 2, 4, 3)
            elif self.dataset == "kitti":
                local_frustums = torch.stack(batch["local_frustums"])
                local_frustums_cnt = torch.stack(batch["local_frustums_cnt"]).float()  # (2, n_frustums, 12)
#                local_frustums = batch["local_frustums"]
#                local_frustums_cnt = torch.stack(batch["local_frustums_cnt"]).float()  # (2, n_frustums, 12)
                n_frustums = local_frustums_cnt.shape[1]
#                print(local_frustums_cnt.shape, n_frustums)

            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = local_frustums_cnt.sum(0) # (n_frustums, n_classes)
#            print(n_frustums)

            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                local_frustum = local_frustums[:, frus, :, :, :].unsqueeze(1).float()
#                local_frustum = local_frustums[:, frus, :, :, :].unsqueeze(1)
#                print(local_frustum.shape, pred_prob.shape)
#                print(local_frustum.type(), pred_prob.type())
                prob = local_frustum * pred_prob # bs, n_classes, 256, 256, 32
                prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.n_classes, -1)
                cum_prob = prob.sum(dim=1) # n_classes 

                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
#                    cum_prob = cum_probs[frus] / total_prob  # (n_classes)
                    cum_prob = cum_prob / total_prob  # (n_classes)
#                    print(frustum_target_proportion.shape, cum_prob.shape)
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion, is_force_empty=False, is_nonzeros=True)
#                    self.log(step_type + 'frustum_loss/frus_'+str(frus), frustum_loss_i.detach(), on_epoch=True, sync_dist=True)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(step_type + '/loss_frustums', frustum_loss.detach(), on_epoch=True, sync_dist=True)

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)
#        self.complet_evaluator.addBatch(y_pred[y_true != 255], y_true[y_true != 255])

#        if 'nonempty' in batch:
        if self.dataset == "NYU":
            tsdf = batch['tsdf_1_4'].cpu().numpy()
            nonempty = (tsdf < 0.1) & (tsdf != 0)
            metric_fov.add_batch(y_pred, y_true, nonempty)
#            fov = torch.stack(batch["valid_pix_{}".format(self.output_scale)]).reshape(bs, 
#                                                            self.output_scene_size[0],
#                                                            self.output_scene_size[2],
#                                                            self.output_scene_size[1])
#            fov = fov.permute(0, 1, 3, 2)
        else:
            fov = torch.stack(batch["valid_pix_{}".format(self.output_scale)]).reshape(bs, 
                                                            self.output_scene_size[0],
                                                            self.output_scene_size[1],
                                                            self.output_scene_size[2])
            fov = fov.detach().cpu().numpy()
            metric_fov.add_batch(y_pred, y_true, fov)

        self.log(step_type + '/loss', loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics, self.train_metrics_fov) 


    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics, self.val_metrics_fov) 

    def validation_epoch_end(self, outputs):
        metric_list = [('train', self.train_metrics), 
                       ('val', self.val_metrics)] 
        metric_list += [('train_fov', self.train_metrics_fov), 
                        ('val_fov', self.val_metrics_fov)]

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
        self.step(batch, "test", self.test_metrics, self.test_metrics_fov) 

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [('test', self.test_metrics), ('test_fov', self.test_metrics_fov)] 
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print("{:.4f}, {:.4f}, {:.4f}".format(stats['precision'] * 100, stats['recall'] * 100, stats['iou'] * 100))
            print("{}, ".format(classes))
            print(' '.join(["{:.4f}, "] * len(classes)).format(*(stats['iou_ssc'] * 100).tolist()))
            print("mIoU", "{:.4f}".format(stats['iou_ssc_mean'] * 100))
            print(stats['iou_ssc_mean'])
            metric.reset()

#        complet_evaluator = self.complet_evaluator
#        _, class_jaccard = complet_evaluator.getIoU()
#        m_jaccard = class_jaccard[1:].mean()
#
#        ignore = [0]
#        # print also classwise
#        for i, jacc in enumerate(class_jaccard):
#            if i not in ignore:
#                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
#                    i=i, class_str=self.class_names[i], jacc=jacc*100))
#
#        # compute remaining metrics.
#        epsilon = np.finfo(np.float32).eps
#        conf = complet_evaluator.get_confusion()
#        precision = np.sum(conf[1:, 1:]) / (np.sum(conf[1:, :]) + epsilon)
#        recall = np.sum(conf[1:, 1:]) / (np.sum(conf[:, 1:]) + epsilon)
#        acc_cmpltn = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0])
#        mIoU_ssc = m_jaccard
#
#        print("Precision =\t" + str(np.round(precision * 100, 2)) + '\n' +
#                   "Recall =\t" + str(np.round(recall * 100, 2)) + '\n' +
#                   "IoU Cmpltn =\t" + str(np.round(acc_cmpltn * 100, 2)) + '\n' +
#                   "mIoU SSC =\t" + str(np.round(mIoU_ssc * 100, 2)))



    def configure_optimizers(self):
        if self.dataset == "NYU":
#            lr = 1e-4
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            print(self.lr, self.weight_decay)
#            return optimizer
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
#            scheduler = CosineAnnealingLR(optimizer, T_max=30)
            return [optimizer], [scheduler]
        elif self.dataset == "kitti":
#            lr = 1e-4
#            print(self.lr, self.weight_decay)
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.1)
            return [optimizer], [scheduler]
#            return optimizer

