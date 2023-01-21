from xmuda.data.semantic_kitti.kitti_dataset import KittiDataset
import numpy as np
import xmuda.common.utils.fusion as fusion
from tqdm import tqdm
from xmuda.data.semantic_kitti.kitti_dm import KittiDataModule
from xmuda.common.utils.sscMetrics import SSCMetrics
from xmuda.data.semantic_kitti.params import kitti_class_names as classes
import os
import cv2
import yaml

kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
kitti_preprocess_root = "/gpfsscratch/rech/kvd/uyl37fq/monoscene_preprocess/kitti"
depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
kitti_tsdf_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"
kitti_seg2d_pcd_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/seg_2d_pcd/kitti"


data_module = KittiDataModule(root=kitti_root,
                        preprocess_root=kitti_preprocess_root,
                        frustum_size=1,
                        n_relations=1,
                        project_scale=2,
                        batch_size=1, 
                        num_workers=1)
data_module.setup()
val_dataset = data_module.val_ds
# train_dataset = KittiDataset(split="train", 
#                              root=kitti_root, 
#                              TSDF_root=kitti_tsdf_root,
#                              depth_root=depth_root)
# val_dataset = KittiDataset(split="val",
#                             root=kitti_root,
#                             TSDF_root=kitti_tsdf_root,
#                             depth_root=depth_root)

# test_dataset = KittiDataset(split="test",
#                             root=kitti_root,
#                             TSDF_root=kitti_tsdf_root,
#                             depth_root=depth_root)



def run(dataset, typ, is_gen_seg_2d=True, is_true_depth=True, scale=4):
    seg_2d_dir = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti/dataset/sequences/{}/gen_2d_seg"
    metric = SSCMetrics(20)
    cnt=0
    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        T_velo_2_cam = batch['T_velo_2_cam']
        P = batch['P']
        frame_id = batch['frame_id']
        sequence = batch['sequence']
        depth_im = batch['depth'][:370, :1220]
        if is_gen_seg_2d:
            seg_2d = np.load(os.path.join(seg_2d_dir.format(sequence), "pred_2d_"+ frame_id + ".png.npy"))
            seg_2d = seg_2d[:370, :1220]
        else:
            seg_2d = np.zeros((370, 1220))

        cam_pts = np.ones((370, 1220, 4))
        cam_pts[:, :, 2] = depth_im

        intr = P[0:3, 0:3]
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix_y, pix_x = np.meshgrid(range(370), range(1220), indexing='ij')
        pix = np.zeros((370, 1220, 2))
        pix[:, :, 0] = pix_x
        pix[:, :, 1] = pix_y
#        print(pix[300, 1200])
        cam_pts[:, :, 0] = (pix[:, :, 0] - cx) * cam_pts[:, :, 2] / fx
        cam_pts[:, :, 1] = (pix[:, :, 1] - cy) * cam_pts[:, :, 2] / fy

        cam_pts = cam_pts.reshape(-1, 4).T
        pcd = np.linalg.inv(T_velo_2_cam) @ cam_pts
        pcd = pcd.T[:, :3]
#        print(np.max(pcd, axis=0))

        pcd[:, 1] = pcd[:, 1] + 25.6
        pcd[:, 2] = pcd[:, 2] + 2
        pcd = pcd / 0.2
        
        save_dir = os.path.join(kitti_seg2d_pcd_root, sequence)
        os.makedirs(save_dir, exist_ok=True)

        # save_path = os.path.join(save_dir, frame_id + ".npy")
#        print(save_path)
        # np.save(save_path, pcd)
        # print("Saved to", save_path)

        y_pred = np.zeros((256, 256, 32))
        pcd = np.round(pcd).astype(int)      
        mask = (pcd[:, 0] >= 0) & (pcd[:, 1] >= 0) & (pcd[:, 2] >= 0) &\
                (pcd[:, 0] <= 255) & (pcd[:, 1] <= 255) & (pcd[:, 2] <= 31)
        


        # config_file = os.path.join('/gpfswork/rech/kvd/uyl37fq/code/xmuda-extend/xmuda/data/semantic_kitti/semantic-kitti.yaml')
        # kitti_config = yaml.safe_load(open(config_file, 'r'))
        # remapdict = kitti_config["learning_map"]
        # maxkey = max(remapdict.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        # remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        # remap_lut[list(remapdict.keys())] = list(remapdict.values())
        # seg_remap_lut = remap_lut - 1
        # seg_remap_lut[seg_remap_lut == -1] = -100

        seg_2d = seg_2d.reshape(-1) + 1
        pcd = pcd[mask]
        seg_2d = seg_2d[mask]
        # print(np.unique(seg_2d))
        # seg_2d = seg_remap_lut[seg_2d]
        y_pred[pcd[:, 0], pcd[:, 1], pcd[:, 2]] = seg_2d

        # print(np.unique(seg_2d))

        y_true = batch['target']
        # mask = (y_true[pcd[:, 0], pcd[:, 1], pcd[:, 2]] == seg_2d)
        # # print(np.sum(mask))
        # pcd_masked = pcd[mask]
        # seg_2d_masked = seg_2d[mask]
        # y_pred[pcd_masked[:, 0], pcd_masked[:, 1], pcd_masked[:, 2]] = seg_2d_masked
        # # print(y_pred.shape, y_true.shape)
        
        metric.add_batch(y_pred, y_true)
        # cnt += 1
        # if cnt == 20:
        #     break   
    
    stats = metric.get_stats()
    print("{:.4f}, {:.4f}, {:.4f}".format(stats['precision'] * 100, stats['recall'] * 100, stats['iou'] * 100))
    print("{}, ".format(classes))
    print(' '.join(["{:.4f}, "] * len(classes)).format(*(stats['iou_ssc'] * 100).tolist()))
    print("mIoU", "{:.4f}".format(stats['iou_ssc_mean'] * 100))
    print(stats['iou_ssc_mean'])
    metric.reset()

is_true_depth = True
#run(train_dataset, "train", is_true_depth=is_true_depth, scale=1)
run(val_dataset, "val", is_true_depth=is_true_depth, scale=1)

#run(test_dataset, "test", is_true_depth=is_true_depth, is_gen_seg_2d=False, scale=1)

                  

