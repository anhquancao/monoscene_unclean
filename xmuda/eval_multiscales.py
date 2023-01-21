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


#model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/no_mask_255/v12_removeCPThreshold_KLnonzeros_LRDecay30_NYU_1_0.0001_0.0001_CPThreshold0.0_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=030-val/mIoU=0.26983.ckpt"
model_path = "/gpfsscratch/rech/kvd/uyl37fq/logs/kitti/v12_ProjectScale2_CPAt1_8_1divlog_LargerFOV_kitti_1_FrusSize_4_WD0_lr0.0001_CEssc_MCAssc_ProportionLoss_CERel_CRCP_Proj_2_4_8/checkpoints/epoch=037-val/mIoU=0.11056.ckpt"

class_weights = {
    '1_4': get_class_weights(NYU_class_freq_1_4).cuda(),
    '1_8': get_class_weights(NYU_class_freq_1_8).cuda(),
    '1_16': get_class_weights(NYU_class_freq_1_16).cuda(),
}


#dataset = "NYU"
dataset = "kitti"
if dataset == "NYU":
    NYU_root = "/gpfswork/rech/kvd/uyl37fq/data/NYU/depthbin"
    NYU_preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
    kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
    full_scene_size = (240, 144, 240)
    output_scene_size = (60, 36, 60)

    NYUdm = NYUDataModule(NYU_root, NYU_preprocess_dir, batch_size=4, num_workers=3)
    NYUdm.setup()
    _C = 12
    data_loader = NYUdm.val_dataloader()
else:
    kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
    kitti_depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
    kitti_logdir = '/gpfsscratch/rech/kvd/uyl37fq/logs/kitti'
    kitti_tsdf_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
    kitti_label_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/labels/kitti"
    kitti_occ_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/occupancy_adabin/kitti"
    kitti_sketch_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/sketch_3D/kitti"
    kitti_mapping_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/mapping_adabin/kitti"
    full_scene_size = (256, 256, 32)
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
    _C = 20
    data_loader = KITTIdm.val_dataloader()

class_relation_weights = get_class_weights(NYU_class_relation_freqs)
model = SSC2dProj3d2d.load_from_checkpoint(model_path)
model.cuda()
model.eval()


count = 0
out_dict = {}
count = 0
write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/draw_output/kitti"
with torch.no_grad():
    for batch in tqdm(data_loader):
        if dataset == "NYU":
            y_true = batch['ssc_label_1_4'].detach().cpu().numpy()
            valid_pix_4 = batch['valid_pix_4']
        else:
            y_true = batch['ssc_label_1_1'].detach().cpu().numpy()
#            valid_pix_1 = batch['valid_pix_1']
            valid_pix_1 = batch['valid_pix_double']

        batch['img'] = batch['img'].cuda()
        pred = model(batch)
        y_pred = torch.softmax(pred['ssc'], dim=1).detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        for i in range(y_true.shape[0]):
            out_dict = {
                "y_pred": y_pred[i].astype(np.uint16),
                "y_true": y_true[i].astype(np.uint16),
            }
            if dataset == "NYU":
                filepath = os.path.join(write_path, batch['name'][i] + ".pkl")
                out_dict["cam_pose"] = batch['cam_pose'][i].detach().cpu().numpy()
                out_dict["vox_origin"] = batch['vox_origin'][i].detach().cpu().numpy()
            elif dataset == "kitti":
                filepath = os.path.join(write_path, batch['sequence'][i], batch['frame_id'][i] + ".pkl")
                out_dict['valid_pix_1'] = valid_pix_1[i].detach().cpu().numpy()
                out_dict['cam_k'] = batch['cam_k'][i].detach().cpu().numpy()
                out_dict['T_velo_2_cam'] = batch['T_velo_2_cam'][i].detach().cpu().numpy()
                os.makedirs(os.path.join(write_path, batch['sequence'][i]), exist_ok=True)

            with open(filepath, 'wb') as handle:
                pickle.dump(out_dict, handle)
                print("wrote to", filepath)

        count += 1
#        if count == 4:
#            break


#    write_path = "/gpfsscratch/rech/kvd/uyl37fq/temp/output"
#    filepath = os.path.join(write_path, "output.pkl")
#    with open(filepath, 'wb') as handle:
#        pickle.dump(out_dict, handle)
#        print("wrote to", filepath)

