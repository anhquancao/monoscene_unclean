from xmuda.models.SSC2d_proj3d2d import SSC2dProj3d2d
from xmuda.data.NYU.nyu_dm_AIC import NYUDataModuleAIC
from xmuda.common.utils.sscMetrics import SSCMetrics
import numpy as np
import torch
from tqdm import tqdm
from xmuda.data.NYU.params import class_weights as NYU_class_weights
#from xmuda.common.utils.metrics import Metrics
from xmuda.models.AIC_trainer import AICTrainer
import pickle
import os

scene_size = (60, 36, 60)
n_classes=12
class_weights = NYU_class_weights 

#preprocess_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti/preprocess/preprocess' 
#model_path = "/gpfsscratch/rech/xqt/uyl37fq/tb_logs/gt_refix_NYU_Noseg2d_SameLR/checkpoints/epoch=80-val_nonempty/mIoU=0.23.ckpt"
#model = SSC2dProj3d2d.load_from_checkpoint(model_path,
#                                           preprocess_dir=preprocess_dir,
#                                           seg_2d=False, 
#                                           input_scene_size=(60, 36, 60), 
#                                           output_scene_size=(60, 36, 60), 
#                                           rgb_encoder="UEffiNet",
#                                           class_weights=class_weights,
#                                           n_classes=n_classes)

model_path = "/gpfsscratch/rech/xqt/uyl37fq/tb_logs/AICnet_SGD_NYU_TrueDepth/checkpoints/epoch=237-val_nonempty/mIoU=0.29.ckpt"
model = AICTrainer.load_from_checkpoint(model_path, 
                                        use_pred_depth=False,
                                        scene_size=scene_size,
                                        n_classes=n_classes,
                                        class_weights=class_weights)

model.cuda()
model.eval()

torch.cuda.empty_cache()

NYU_root = "/gpfswork/rech/xqt/uyl37fq/data/NYU/depthbin"
NYU_preprocess_dir = "/gpfsscratch/rech/xqt/uyl37fq/precompute_data/NYU_AIC"
pred_depth_dir = "/gpfsscratch/rech/xqt/uyl37fq/NYU_pred_depth"

NYUdm = NYUDataModuleAIC(NYU_root, NYU_preprocess_dir, pred_depth_dir, batch_size=4)
NYUdm.setup()

_C = 12
count = 0
#metrics = Metrics(_C)
tp, fp, fn = np.zeros(_C, dtype=np.int32), np.zeros(_C, dtype=np.int32), np.zeros(_C, dtype=np.int32)
sscMetrics = SSCMetrics(_C)
sscMetrics_nonempty = SSCMetrics(_C)
predict_empty_using_TSDF = True

def get_nonempty(voxels, encoding):  # Get none empty from depth voxels
    data = np.zeros(voxels.shape, dtype=np.float32)  # init 0 for empty
    # if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
    #     data[voxels == 1] = 1
    #     return data
    if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
        data[voxels != 0] = 1
        surface = np.array(np.where(voxels == 1))  # surface=1
    elif encoding == 'TSDF':
        data[np.where(np.logical_or(voxels <= 0, voxels == 1))] = 1
        surface = np.array(np.where(voxels == 1))  # surface
        # surface = np.array(np.where(np.logical_and(voxels > 0, voxels != np.float32(0.001))))  # surface
    else:
        raise Exception("Encoding error: {} is not validate".format(encoding))

    min_idx = np.amin(surface, axis=1)
    max_idx = np.amax(surface, axis=1)
    # print('min_idx, max_idx', min_idx, max_idx)
    # data[:a], data[a]不包含在内, data[b:], data[b]包含在内
    # min_idx = min_idx
    max_idx = max_idx + 1
    # 本该扩大一圈就够了，但由于GT标注的不是很精确，故在高分辨率情况下，多加大一圈
    # min_idx = min_idx - 1
    # max_idx = max_idx + 2
    min_idx[min_idx < 0] = 0
    max_idx[0] = min(voxels.shape[0], max_idx[0])
    max_idx[1] = min(voxels.shape[1], max_idx[1])
    max_idx[2] = min(voxels.shape[2], max_idx[2])
    data[:min_idx[0], :, :] = 0  # data[:a], data[a]不包含在内
    data[:, :min_idx[1], :] = 0
    data[:, :, :min_idx[2]] = 0
    data[max_idx[0]:, :, :] = 0  # data[b:], data[b]包含在内
    data[:, max_idx[1]:, :] = 0
    data[:, :, max_idx[2]:] = 0
    return data

out_dict = {}
with torch.no_grad():
    for batch in tqdm(NYUdm.val_dataloader()):
#        rgb, depth, volume, y_true, nonempty, position, filename = batch
#        model.validation_step(batch, 0) 
        y_true = batch['ssc_label_1_4'].detach().cpu().numpy()
        nonempty = batch['nonempty']
        tsdf = batch['tsdf_1_4']
#        nonempty = get_nonempty(tsdf, 'TSDF')

        out = model(batch)
        ssc_logit = out['ssc_logit']
#        tsdf_1_4_pred = out['tsdf_1_4'].squeeze().detach().cpu().numpy()
#        print(tsdf_1_4_pred)
        y_pred = out['ssc_logit'].detach().cpu().numpy()
        print(y_pred.shape)
        y_pred = np.argmax(y_pred, axis=1)

        for i in range(y_pred.shape[0]):
            out_dict[batch['scene'][i]] = {
                "y_pred": y_pred[i],
                "y_true": y_true[i],
                "tsdf": tsdf[i].cpu().numpy()

            }
#        y_pred[(tsdf_1_4_pred > 0.1) & (tsdf_1_4_pred < 0.8)] = 0
#        print(nonempty.shape, y_pred.shape)
        if predict_empty_using_TSDF:
            y_pred[nonempty == 0] = 0     # 0 empty
#        metrics.add_batch(prediction=ssc_logit, target=y_true)

        sscMetrics.add_batch(y_pred, y_true)
        sscMetrics_nonempty.add_batch(y_pred, y_true, nonempty)


    write_path = "/gpfsscratch/rech/xqt/uyl37fq/temp/output"
    filepath = os.path.join(write_path, "out.pkl")
    with open(filepath, 'wb') as handle:
        pickle.dump(out_dict, handle)
        print("wrote to", filepath)

    print("nonempty = None")
    stats = sscMetrics.get_stats()
    print(stats)
    print("================")
    print("nonempty != None")
    stats_nonempty = sscMetrics_nonempty.get_stats()
    print(stats_nonempty)
    #val_p = stats['precision']
    #val_r = stats['recall']
    #val_iou = stats['iou']
    #val_iou_ssc = stats['iou_ssc']
    #val_iou_ssc_mean = stats['iou_ssc_mean']
    #
    #print('Validate with TSDF:, p {:.1f}, r {:.1f}, IoU {:.1f}'.format(val_p*100.0, val_r*100.0, val_iou*100.0))
    #print('pixel-acc {:.4f}, mean IoU {:.1f}, SSC IoU:{}'.format(val_acc*100.0, val_iou_ssc_mean*100.0, val_iou_ssc*100.0))
    #print("=============")
    #print('Validate with TSDF:, p {:.1f}, r {:.1f}, IoU {:.1f}'.format(
    #    metrics.get_occupancy_Precision()*100.0, 
    #    metrics.get_occupancy_Recall()*100.0, 
    #    metrics.get_occupancy_IoU()*100.0))
    #print('pixel-acc {:.4f}, mean IoU {:.1f}, SSC IoU:{}'.format(
    #    -1, metrics.get_semantics_mIoU()*100.0, -1))
