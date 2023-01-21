import pytorch_lightning as pl
import torch
import torch.nn as nn
from xmuda.common.utils.metrics import Metrics
import xmuda.data.semantic_kitti.io_data as SemanticKittiIO
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.models.sketch_init import init_weight, group_weight
import pickle
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
from xmuda.models.sketch import Sketch3DSSC


class SketchTrainer(pl.LightningModule):
    def __init__(self,
                 n_classes,
                 class_names,
                 class_weights,
                 full_scene_size,
                 output_scene_size,
                 n_training_items,
                 dataset,
                 predict_empty_from_depth=False,
                 optimize_everywhere=False,
                 use_3DSketch_nonempty_mask=True,
                 voxel_size=0.02,
                 lr=1e-4):
        super().__init__()
        self.predict_empty_from_depth = predict_empty_from_depth
        self.optimize_everywhere = optimize_everywhere
        self.use_3DSketch_nonempty_mask = use_3DSketch_nonempty_mask
        self.class_names = class_names
        self.full_scene_size = full_scene_size
        self.output_scene_size = output_scene_size
        self.dataset = dataset

        self.voxel_size = voxel_size
        self.n_classes = n_classes

        self.class_weights = class_weights
        self.lr = lr

        # log hyperparameters
        self.save_hyperparameters()

        self.train_metric = SSCMetrics(self.n_classes)
        self.train_metric_nonempty = SSCMetrics(self.n_classes)
        self.train_metric_prednonempty = SSCMetrics(self.n_classes)
        self.val_metric = SSCMetrics(self.n_classes)
        self.val_metric_nonempty = SSCMetrics(self.n_classes)
        self.val_metric_prednonempty = SSCMetrics(self.n_classes)
        self.test_metric = SSCMetrics(self.n_classes)
        self.test_metric_nonempty = SSCMetrics(self.n_classes)
        self.test_metric_prednonempty = SSCMetrics(self.n_classes)

        resnet_weight_path = "/gpfswork/rech/kvd/uyl37fq/code/xmuda-extend/xmuda/weights/resnet50.pth"
        if self.output_scene_size[0] == self.full_scene_size[0]:
#            feature = 64
#            feature_oper = 32
            feature = 128
            feature_oper = 64
        else:
            feature = 128
            feature_oper = 64
        self.model = Sketch3DSSC(self.n_classes, base_lr=self.lr, 
                                 full_scene_size=full_scene_size,
                                 output_scene_size=output_scene_size,
                                 optimize_everywhere=self.optimize_everywhere,
                                 feature_oper=feature_oper,
                                 feature=feature, bn_momentum=0.1, pretrained_model=resnet_weight_path) 

        self.bn_eps = 1e-5
        self.bn_momentum = 0.9
        init_weight(self.model.business_layer, nn.init.kaiming_normal_, 
                    self.bn_eps, self.bn_momentum, mode='fan_in')

        ''' Intialize state dict resnet50 '''
        state_dict = torch.load(resnet_weight_path)
        transformed_state_dict = {}
        for k, v in state_dict.items():
            transformed_state_dict[k.replace('.bn.', '.')] = v

        self.model.backbone.load_state_dict(transformed_state_dict, strict=False)

        ''' fix the weight of resnet '''
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        self.base_lr = 0.1
        self.n_iters_per_epoch = n_training_items // 4
        self.total_iters = 250 * self.n_iters_per_epoch 
        self.lr_power = 0.9
#        self.cur_train_idx = 0


    def forward(self, batch):
        if 'sketch_1_4' in batch:
            sketch = batch['sketch_1_4']
        else:
            sketch = None
        input_data = {
            'img': batch['img'], 
            'tsdf': batch['tsdf_1_4'],
            'MAPPING_2DRGB_3DGRID': batch['mapping_1_4'],
            '3D_SKETCH': sketch
        }
        out = self.model(input_data) 
        out['ssc_logit'] = out['pred_semantic']
        return out 

    def step(self, batch, step_type, metric, metric_nonempty, metric_prednonempty):
        scores = self(batch)

        if self.dataset == "kitti" and step_type == "test":
            y_pred = scores['pred_semantic'].detach().cpu().numpy()
            inv_map = SemanticKittiIO.get_inv_map()
            root_path = '/gpfsscratch/rech/kvd/uyl37fq/test_pred/3DSketch/sequences/{}/predictions'
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
            y_pred = scores['pred_semantic'].detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            tsdf = batch['tsdf_1_4'].cpu().numpy()
#            if self.predict_empty_from_depth:
#                pred_nonempty = (tsdf < 0.1) & (tsdf != 0)
#                y_pred[pred_nonempty == 0] = 0 
#                metric_prednonempty.add_batch(y_pred, y_true, pred_nonempty, nonsurface)
            if self.output_scene_size[0] == self.full_scene_size[0]:
                y_true = batch['target']
            else:
                y_true = batch['ssc_label_1_4']
            loss = self.model.compute_loss(scores, y_true, batch,
                                           self.class_weights.type_as(scores['pred_semantic']),
                                           use_3DSketch_nonempty_mask=self.use_3DSketch_nonempty_mask)['total']

            self.log(step_type + '/loss', loss.item(), on_epoch=True, sync_dist=True)
            y_true = y_true.cpu().numpy()
            metric.add_batch(y_pred, y_true)

            nonempty = (tsdf < 0.1) & (tsdf != 0)
#            y_pred[~nonempty] = 0
            metric_nonempty.add_batch(y_pred, y_true, nonempty)
    #        if 'sketch_original_mapping' in batch:
    #            nonsurface = batch['sketch_original_mapping'].cpu().numpy() == 307200
    #            nonempty = batch['nonempty'].cpu().numpy()
    #            metric_nonempty.add_batch(y_pred, y_true, nonempty, nonsurface)

            return loss

    def training_step(self, batch, batch_idx):
        lr = self.poly_lr(self.global_step)
#        print(self.global_step, lr)
#        lr = self.poly_lr(self.cur_train_idx)
#        self.cur_train_idx += 1
        optimizer = self.optimizers()
        for group in optimizer.param_groups: 
            group["lr"] = lr
        self.log("train/lr", lr, on_epoch=False, on_step=True, sync_dist=True)

        loss = self.step(batch, "train", self.train_metric, self.train_metric_nonempty, self.train_metric_prednonempty)
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metric, self.val_metric_nonempty, self.val_metric_prednonempty)

    def validation_epoch_end(self, outputs):

        metric_list = [('train', self.train_metric), ('val', self.val_metric)] 
        metric_list += [('train_nonempty', self.train_metric_nonempty), ('val_nonempty', self.val_metric_nonempty)]
        metric_list += [('train_prednonempty', self.train_metric_prednonempty), ('val_prednonempty', self.val_metric_prednonempty)]

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
        self.step(batch, "test", self.test_metric, self.test_metric_nonempty, self.test_metric_prednonempty)

    def test_epoch_end(self, outputs):
        metric_list = [('test', self.test_metric), ('test_nonempty', self.test_metric_nonempty)] 
#        metric_list += [('test_prednonempty', self.test_metric_prednonempty)]

        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print("{:.4f}, {:.4f}, {:.4f}".format(stats['precision'] * 100, stats['recall'] * 100, stats['iou'] * 100))
            print("{}, ".format(self.class_names))
            print(' '.join(["{:.4f}, "] * len(self.class_names)).format(*(stats['iou_ssc'] * 100).tolist()))
            print("mIoU", "{:.4f}".format(stats['iou_ssc_mean'] * 100))
            metric.reset()

    def poly_lr(self, cur_iter):
       return self.base_lr * ((1 - float(cur_iter) / self.total_iters) ** self.lr_power)


    def configure_optimizers(self):
        params_list = []
        for module in self.model.business_layer:
            params_list = group_weight(params_list, module, self.base_lr)
        params_list = self.model.parameters()
        optimizer = torch.optim.SGD(params_list, 
                                    lr=self.base_lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
        return optimizer
