from xmuda.models.SSC2d_proj3d2d import SSC2dProj3d2d
from xmuda.data.NYU.nyu_dm import NYUDataModule
from xmuda.models.SketchTrainer import SketchTrainer
from xmuda.models.LMSC_trainer import LMSCTrainer
from xmuda.models.AIC_trainer import AICTrainer
from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.data.semantic_kitti.params import semantic_kitti_class_frequencies, kitti_class_names, class_weights as kitti_class_weights
from xmuda.data.NYU.params import class_relation_freqs as NYU_class_relation_freqs, class_weights as NYU_class_weights, class_freq_1_4 as NYU_class_freq_1_4, class_freq_1_8 as NYU_class_freq_1_8, class_freq_1_16 as NYU_class_freq_1_16, class_relation_weights as NYU_class_relation_weights, NYU_class_names
import numpy as np
import torch
import torch.nn.functional as F
from xmuda.models.ssc_loss import get_class_weights
from tqdm import tqdm
import pickle
import os


#model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_new/v22_full_NYU_3_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=022-val/mIoU=0.27023.ckpt"
#model_path_1 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=027-val/mIoU=0.10886.ckpt"
model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=029-val/mIoU=0.11596.ckpt"
model_path_1 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_1_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=027-val/mIoU=0.11444.ckpt"
model_path_2 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_3_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=026-val/mIoU=0.11318.ckpt"

class_weights = {
    '1_4': get_class_weights(NYU_class_freq_1_4).cuda(),
    '1_8': get_class_weights(NYU_class_freq_1_8).cuda(),
    '1_16': get_class_weights(NYU_class_freq_1_16).cuda(),
}


#dataset = "NYU"
dataset = "kitti"
if dataset == "NYU":
    AIC_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines/baseline_1_1_1divlogLabelWeights_FixOptimizer_AICNet_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=197-val/mIoU=0.175.ckpt"
    sketch_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines/baseline_1_1_1divlogLabelWeights_FixOptimizer_3DSketch_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=241-val/mIoU=0.226.ckpt"
    lmscnet_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines/baseline_1_1_1divlogLabelWeights_FixOptimizer_LMSCNet_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=48-val/mIoU=0.157.ckpt"
    lmsc_channels = 144
    NYU_root = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin"
    NYU_preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
    full_scene_size = (240, 144, 240)
    output_scene_size = (60, 36, 60)
    class_names = NYU_class_names 
    class_weights = NYU_class_weights

    NYUdm = NYUDataModule(NYU_root, NYU_preprocess_dir, batch_size=4, 
                          use_predicted_depth=True,
                          num_workers=3)
    NYUdm.setup()
    n_classes = 12
    data_loader = NYUdm.val_dataloader()

    class_relation_weights = get_class_weights(NYU_class_relation_freqs)

else:
    AIC_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_kitti/baseline_1_1_AICNet_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=64-val/mIoU=0.081.ckpt"
    sketch_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_kitti/baseline_1_1_3DSketch_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=66-val/mIoU=0.074.ckpt"
    lmscnet_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_kitti/baseline_1_1_1divlogLabelWeights_FixOptimizer_LMSCNet_kitti_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=84-val/mIoU=0.082.ckpt"
    lmsc_channels = 32
    kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
    kitti_depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
    kitti_logdir = '/gpfsscratch/rech/kvd/uyl37fq/logs/kitti'
    kitti_tsdf_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
    kitti_label_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/labels/kitti"
    kitti_occ_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/occupancy_adabin/kitti"
    kitti_sketch_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/sketch_3D/kitti"
    kitti_mapping_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/mapping_adabin/kitti"
    full_scene_size = (256, 256, 32)
    output_scene_size = full_scene_size
    class_names = kitti_class_names
    epsilon_w = 0.001  # eps to avoid zero division
    class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05, 6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05, 2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07, 2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08, 2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
    class_weights = torch.from_numpy(1 / np.log(class_frequencies + epsilon_w))
    KITTIdm = KittiDataModule(root=kitti_root,
                              data_aug=True,
                              TSDF_root=kitti_tsdf_root,
                              label_root=kitti_label_root,
                              mapping_root=kitti_mapping_root,
                              occ_root=kitti_occ_root,
                              depth_root=kitti_depth_root,
                              sketch_root=kitti_sketch_root,
                              batch_size=1,
                              num_workers=3)
    KITTIdm.setup()
    n_classes = 20
    data_loader = KITTIdm.val_dataloader()


aicnet = AICTrainer.load_from_checkpoint(AIC_path ,
                                         n_classes=n_classes,
                                         dataset=dataset,
                                         full_scene_size=full_scene_size,
                                         output_scene_size=output_scene_size,
                                         class_weights=class_weights,
                                         class_names=class_names)
aicnet.cuda()
aicnet.eval()
sketch = SketchTrainer.load_from_checkpoint(sketch_path, 
                                          n_classes=n_classes, 
                                          dataset=dataset, 
                                          full_scene_size=full_scene_size, 
                                          output_scene_size=output_scene_size, 
                                          class_names=class_names)
sketch.cuda()
sketch.eval()
lmscnet = LMSCTrainer.load_from_checkpoint(lmscnet_path, 
                                           n_classes=n_classes, 
                                           dataset=dataset, 
                                           in_channels=lmsc_channels,
                                           full_scene_size=full_scene_size, 
                                           output_scene_size=output_scene_size, 
                                           class_names=class_names)
lmscnet.cuda()
lmscnet.eval()

our = SSC2dProj3d2d.load_from_checkpoint(model_path)
our.cuda()
our.eval()
our1 = SSC2dProj3d2d.load_from_checkpoint(model_path_1)
our1.cuda()
our1.eval()
our2 = SSC2dProj3d2d.load_from_checkpoint(model_path_2)
our2.cuda()
our2.eval()

#others = [
#    our, our1, our2, lmscnet, aicnet, sketch
#]
others = [
     sketch
]


count = 0
out_dict = {}
count = 0
#write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/draw_output/{}".format(dataset)
write_path = "/gpfswork/rech/kvd/uyl37fq/code/temp/draw_output/{}".format(dataset)
if dataset == "NYU":
    other_metrics = [SSCMetrics(n_classes) for i in range(len(others))]
else:
    other_metrics = [SSCMetrics(n_classes) for i in range(len(others) + 1)]
    other_metrics_fovs = [SSCMetrics(n_classes) for i in range(len(others) + 1)]
    classes = ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 
                   'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 
                   'road', 'parking', 'sidewalk', 'other-ground', 'building', 
                   'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
our_metric = SSCMetrics(n_classes)
our_metric_fov = SSCMetrics(n_classes)
cnt = 0

with torch.no_grad():
    for batch in tqdm(data_loader):
#        cnt += 1
#        if cnt == 10:
#            break
        if dataset == "NYU":
            y_true = batch['ssc_label_1_4'].detach().cpu().numpy()
            valid_pix_4 = batch['valid_pix_4']
        else:
            y_true = batch['target'].detach().cpu().numpy()
            valid_pix_1 = batch['valid_pix_1']
            frame_ids = batch['frame_id']
            js3c_preds = []
#            js3d_pred_path = "/gpfswork/rech/kvd/uyl37fq/code/JS3C-Net/log/sem_pcd_v2/dump/completion/submit_valid2021_11_06/sequences/08/predictions/{}.label"
#            js3d_pred_path = "/gpfswork/rech/kvd/uyl37fq/code/JS3C-original/log/JS3C-Net-kitti/dump/completion/submit_valid2021_11_13/sequences/08/predictions/{}.label"
            js3d_pred_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/LMSCNet_3D/sequences/08/predicitons/{}.label"
            for i in range(y_true.shape[0]):
                js3c_pred = np.fromfile(js3d_pred_path.format(frame_ids[i]), dtype=np.uint16).reshape(1, 256, 256, 32)
                js3c_preds.append(js3c_pred)
            js3c_preds = np.concatenate(js3c_preds, axis=0)

        for key in ['img', 'mapping_1_1','depth', 'mapping_1_4', 'tsdf_1_4']:
            batch[key] = batch[key].cuda()
#        pred = np.argmax(our(batch)['ssc'].detach().cpu().numpy(), axis=1)
        other_preds = [np.argmax(m(batch)['ssc_logit'].detach().cpu().numpy(), axis=1) for m in others]
        other_preds.append(js3c_preds)
        for i in range(y_true.shape[0]):
#            our_metric.add_batch(pred[i], y_true[i])
#            our_metric_fov.add_batch(pred[i], y_true[i], ~valid_pix_1[0].detach().cpu().numpy().reshape(256, 256, 32))
            for j, other_metric in enumerate(other_metrics):
                other_metric.add_batch(other_preds[j][i], y_true[i])
                other_metrics_fovs[j].add_batch(other_preds[j][i], y_true[i], valid_pix_1[0].detach().cpu().numpy().reshape(256, 256, 32))
#    methods = ["our", "our1", "our2", 'lmscnet', 'aicnet', 'sketch', "js3c_net"]
    methods = ['sketch', "js3c_net"]

#    metric_list = [('our', our_metric), ('our_fov', our_metric_fov)]
    metric_list = []
    for k in range(len(methods)):
        metric_list.append((methods[k], other_metrics[k]))
        metric_list.append((methods[k] + "_fov", other_metrics_fovs[k]))
    for prefix, metric in metric_list:
        print("{}==========".format(prefix))
        stats = metric.get_stats()
        print("{:.4f}, {:.4f}, {:.4f}".format(stats['precision'] * 100, stats['recall'] * 100, stats['iou'] * 100))
        print("{}, ".format(classes))
        print(' '.join(["{:.4f}, "] * len(classes)).format(*(stats['iou_ssc'] * 100).tolist()))
        print("mIoU", "{:.4f}".format(stats['iou_ssc_mean'] * 100))
        print(stats['iou_ssc_mean'])
        metric.reset()
