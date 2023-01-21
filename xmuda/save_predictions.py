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


model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/NYU_new/v22_full_NYU_3_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=022-val/mIoU=0.27023.ckpt"
#model_path_1 = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=027-val/mIoU=0.10886.ckpt"
#model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti_ablate_1_1/FullRes_kitti_2_FrusSize_8_nRelations4_optimizeIoUTrue_lovaszFalse_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=029-val/mIoU=0.11596.ckpt"
#
class_weights = {
    '1_4': get_class_weights(NYU_class_freq_1_4).cuda(),
    '1_8': get_class_weights(NYU_class_freq_1_8).cuda(),
    '1_16': get_class_weights(NYU_class_freq_1_16).cuda(),
}


dataset = "NYU"
#dataset = "kitti"
if dataset == "NYU":
    AIC_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_1divlogLabelWeights_FixOptimizer_AICNet_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=197-val/mIoU=0.175.ckpt"
    sketch_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_1divlogLabelWeights_FixOptimizer_3DSketch_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=241-val/mIoU=0.226.ckpt"
    lmscnet_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_1divlogLabelWeights_FixOptimizer_LMSCNet_NYU_PredDepthTrue_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=48-val/mIoU=0.157.ckpt"
#    AIC_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_AICNet_NYU_PredDepthFalse_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=249-val/mIoU=0.238.ckpt"
#    sketch_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_3DSketch_NYU_PredDepthFalse_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=221-val/mIoU=0.292.ckpt"
#    lmscnet_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/baselines_NYU/baseline_1_1_LMSCNet_NYU_PredDepthFalse_OptimizeEverywhereTrue_3DSketchNonemptyTrue_EmptyFromDepthFalse/checkpoints/epoch=43-val/mIoU=0.205.ckpt"
    lmsc_channels = 144
    NYU_root = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin"
    NYU_preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
    full_scene_size = (240, 144, 240)
    output_scene_size = (60, 36, 60)
    class_names = NYU_class_names 
    class_weights = NYU_class_weights

    NYUdm = NYUDataModule(NYU_root, NYU_preprocess_dir, batch_size=4, 
                          use_predicted_depth=True,
#                          use_predicted_depth=False,
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
    kitti_preprocess_root = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/kitti"
#    kitti_depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
#    kitti_logdir = '/gpfsscratch/rech/kvd/uyl37fq/logs/kitti'
#    kitti_tsdf_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
#    kitti_label_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/labels/kitti"
#    kitti_occ_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/occupancy_adabin/kitti"
#    kitti_sketch_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/sketch_3D/kitti"
#    kitti_mapping_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/mapping_adabin/kitti"
    full_scene_size = (256, 256, 32)
    output_scene_size = full_scene_size
    class_names = kitti_class_names
    epsilon_w = 0.001  # eps to avoid zero division
    class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05, 6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05, 2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07, 2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08, 2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])
    class_weights = torch.from_numpy(1 / np.log(class_frequencies + epsilon_w))
    KITTIdm = KittiDataModule(root=kitti_root,
                              preprocess_root=kitti_preprocess_root,
#                              TSDF_root=kitti_tsdf_root,
#                              label_root=kitti_label_root,
#                              mapping_root=kitti_mapping_root,
#                              occ_root=kitti_occ_root,
#                              depth_root=kitti_depth_root,
#                              sketch_root=kitti_sketch_root,
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
others = [
    lmscnet, aicnet, sketch
]

#others = []

our = SSC2dProj3d2d.load_from_checkpoint(model_path)
our.cuda()
our.eval()

count = 0
out_dict = {}
count = 0
write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/draw_output/{}_all_rgb_baselines".format(dataset)
#write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/draw_output/{}_all_3d_baselines".format(dataset)
#write_path = "/gpfswork/rech/kvd/uyl37fq/code/temp/draw_output/kitti_js3cnet_3d"
if dataset == "NYU":
    other_metrics = [SSCMetrics(n_classes) for i in range(len(others))]
else:
    other_metrics = [SSCMetrics(n_classes) for i in range(len(others) + 1)]
our_metric = SSCMetrics(n_classes)
our_metric_fov = SSCMetrics(n_classes)
our_metric_outfov = SSCMetrics(n_classes)
cnt = 0

with torch.no_grad():
    for batch in tqdm(data_loader):
#        if batch['frame_id'][0] not in ["001385", "000295", "002505", "001005"]:
#            continue
#        cnt += 1
#        if cnt < 523:
#            continue
        if dataset == "NYU":
            y_true = batch['ssc_label_1_4'].detach().cpu().numpy()
            valid_pix_4 = batch['valid_pix_4']
        else:
            y_true = batch['target'].detach().cpu().numpy()
            valid_pix_1 = batch['valid_pix_1']
            frame_ids = batch['frame_id']
#            sequences = batch['sequence']
            js3c_preds = []
#            js3d_pred_path = "/gpfswork/rech/kvd/uyl37fq/code/JS3C-Net/log/sem_pcd_v2/dump/completion/submit_valid2021_11_06/sequences/08/predictions/{}.label"
            js3d_pred_path = "/gpfswork/rech/kvd/uyl37fq/code/js3cnet_original/log/JS3C-Net-kitti/dump/completion/submit_valid2021_11_20/sequences/08/predictions/{}.label"
            for i in range(y_true.shape[0]):
                js3c_pred = np.fromfile(js3d_pred_path.format(frame_ids[i]), dtype=np.uint16).reshape(1, 256, 256, 32)
#                print(y_true[i].shape, js3c_pred.shape)
                js3c_preds.append(js3c_pred)
            js3c_preds = np.concatenate(js3c_preds, axis=0)

        for key in ['img', 'mapping_1_1','depth', 'mapping_1_4', 'tsdf_1_4']:
            batch[key] = batch[key].cuda()
        pred = np.argmax(our(batch)['ssc'].detach().cpu().numpy(), axis=1)
        other_preds = [np.argmax(m(batch)['ssc_logit'].detach().cpu().numpy(), axis=1) for m in others]
        if dataset == "kitti":
            other_preds.append(js3c_preds)
        for i in range(y_true.shape[0]):
            our_metric.reset()
            for metric in other_metrics:
                metric.reset()
            our_metric.add_batch(pred[i], y_true[i])
            mIoU = our_metric.get_stats()['iou_ssc_mean'] * 100
            if dataset == "kitti":
                our_metric_outfov.reset()
                our_metric_fov.reset()
                our_metric_outfov.add_batch(pred[i], y_true[i], valid_pix_1[0].detach().cpu().numpy().reshape(256, 256, 32))
                our_metric_fov.add_batch(pred[i], y_true[i], valid_pix_1[0].detach().cpu().numpy().reshape(256, 256, 32))
                fov_mIoU = our_metric_fov.get_stats()['iou_ssc_mean'] * 100
                outfov_mIoU = our_metric_outfov.get_stats()['iou_ssc_mean'] * 100
            other_mIoUs = []
            for j, other_metric in enumerate(other_metrics):
                other_metric.add_batch(other_preds[j][i], y_true[i])
                other_mIoUs.append(other_metric.get_stats()['iou_ssc_mean'] * 100)
            quality = mIoU - np.max(other_mIoUs)
#            quality = mIoU
#            if quality > 2:
            classes = np.unique(y_true[i])
            classes_in_scene = len(classes)
            out_dict = {
                "y_preds": [other_pred[i].astype(np.uint16) for other_pred in other_preds],
                "our_pred": pred[i].astype(np.uint16),
                "y_true": y_true[i].astype(np.uint16)
            }
            if dataset == "NYU":
                filepath = os.path.join(write_path, batch['name'][i] + "_quality={:.4f}_nclasses={}.pkl".format(quality, classes_in_scene))
                out_dict["cam_pose"] = batch['cam_pose'][i].detach().cpu().numpy()
                out_dict["vox_origin"] = batch['vox_origin'][i].detach().cpu().numpy()
                os.makedirs(write_path, exist_ok=True)
            else:
                filepath = os.path.join(write_path, batch['frame_id'][i] + "_quality={:.4f}_nclasses={}_fov{:.4f}_outfov{:.4f}.pkl".format(quality,
                                                                                                                                   classes_in_scene,
                                                                                                                                   fov_mIoU,
                                                                                                                                   outfov_mIoU))
                out_dict["valid_pix_1"] = batch['valid_pix_1'][i].detach().cpu().numpy()
                out_dict["cam_k"] = batch['cam_k'][i].detach().cpu().numpy()
                out_dict["T_velo_2_cam"] = batch['T_velo_2_cam'][i].detach().cpu().numpy()
                os.makedirs(write_path, exist_ok=True)
            with open(filepath, 'wb') as handle:
                print(list(out_dict.keys()))
                pickle.dump(out_dict, handle)
                print("wrote to", filepath)

