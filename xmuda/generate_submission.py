from xmuda.models.SSC2d_proj3d2d import SSC2dProj3d2d
from xmuda.data.NYU.nyu_dm import NYUDataModule
from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.data.NYU.params import class_relation_freqs as NYU_class_relation_freqs, class_freq_1_4 as NYU_class_freq_1_4, class_freq_1_8 as NYU_class_freq_1_8, class_freq_1_16 as NYU_class_freq_1_16
import numpy as np
import torch
import torch.nn.functional as F
from xmuda.models.ssc_loss import get_class_weights
from tqdm import tqdm
import pickle
import os

model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/no_mask_255/v9_MegaCP_NoRelAffLoss_FixedThreshold_AddFrustumLoss_KLSep_NYU_1_0.0001_0.0001_1_1_EmptyMul1.0_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=040-val/mIoU=0.25968.ckpt"

class_weights = {
    '1_4': get_class_weights(NYU_class_freq_1_4).cuda(),
    '1_8': get_class_weights(NYU_class_freq_1_8).cuda(),
    '1_16': get_class_weights(NYU_class_freq_1_16).cuda(),
}


dataset = "NYU"
kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
kitti_depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
kitti_logdir = '/gpfsscratch/rech/kvd/uyl37fq/logs/kitti'
kitti_tsdf_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
kitti_label_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/labels/kitti"
kitti_occ_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/occupancy_adabin/kitti"
kitti_sketch_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/sketch_3D/kitti"
kitti_mapping_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/mapping_adabin/kitti"
full_scene_size = (256, 256, 32)
output_scene_size = (256 // 4, 256 // 4, 32 // 4)
KITTIdm = KittiDataModule(root=kitti_root,
                          data_aug=True,
                          TSDF_root=kitti_tsdf_root,
                          label_root=kitti_label_root,
                          mapping_root=kitti_mapping_root,
                          occ_root=kitti_occ_root,
                          depth_root=kitti_depth_root,
                          sketch_root=kitti_sketch_root,
                          batch_size=4,
                          num_workers=3)
KITTIdm.setup()
_C = 20
data_loader = KITTIdm.val_dataloader()

class_relation_weights = get_class_weights(NYU_class_relation_freqs)
model = SSC2dProj3d2d.load_from_checkpoint(model_path, 
                                           output_scene_size=output_scene_size,
                                           full_scene_size=full_scene_size,
                                           use_class_proportion=False, 
                                           class_proportion_loss=True)
model.cuda()
model.eval()
#model_KL = SSC2dProj3d2d.load_from_checkpoint(model_KL_path, 
#                                              output_scene_size=output_scene_size,
#                                              full_scene_size=full_scene_size,
#                                              use_class_proportion=False, 
#                                              class_proportion_loss=False)
#model_KL.cuda()
#model_KL.eval()


count = 0
out_dict = {}
count = 0
with torch.no_grad():
    for batch in tqdm(data_loader):
        y_true = batch['ssc_label_1_4'].detach().cpu().numpy()

        batch['img'] = batch['img'].cuda()
        pred = model(batch, max_k=128)
        y_pred = torch.softmax(pred['ssc'], dim=1).detach().cpu().numpy()
#        pred_KL = model_KL(batch, max_k=128)
#        y_pred_KL = pred_KL['ssc'].detach().cpu().numpy()

#        y_pred = np.argmax(y_pred, axis=1)
        for i in range(y_true.shape[0]):
            out_dict[batch['scene'][i]] = {
#                "P_logits": torch.sigmoid(pred['P_logits'][i]).detach().cpu().numpy(),
#                "CP_mega_matrices": batch['CP_mega_matrices'][i]
                "y_pred": y_pred[i],
#                "y_pred_KL": y_pred_KL[i],
                "y_true": y_true[i],
#model_KL = SSC2dProj3d2d.load_from_checkpoint(model_KL_path, 
#                                              output_scene_size=output_scene_size,
#                                              full_scene_size=full_scene_size,
#                                              use_class_proportion=False, 
#                                              class_proportion_loss=False)
#model_KL.cuda()
#model_KL.eval()
#                "local_frustums": batch['local_frustums_4'][i].detach().cpu().numpy(),
#                "sketch": batch['sketch'][i],
#                "y_true_1_16": y_true_1_16[i],
#                "nonempty": batch['nonempty'][i],
#                "tsdf": batch['sketch_tsdf'][i],
#                "keep_idx_1_4": batch['keep_idx_1_4'][i],
#                "tsdf": tsdf[i].cpu().numpy(),
#                "relate_prob": relate_probs[i].detach().cpu().numpy(),
#                "cam_pose": batch['cam_pose'][i],
#                "vox_origin": batch['vox_origin'][i]
            }

        count += 1
        if count == 4:
            break


    write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/output"
    filepath = os.path.join(write_path, "relation_0.pkl")
    with open(filepath, 'wb') as handle:
        pickle.dump(out_dict, handle)
        print("wrote to", filepath)

