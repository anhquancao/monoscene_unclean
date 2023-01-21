from xmuda.data.semantic_kitti.kitti_dataset import KittiDataset
import numpy as np
import xmuda.common.utils.fusion as fusion
from tqdm import tqdm
import os
import cv2

kitti_root = "/gpfswork/rech/kvd/uyl37fq/data/semantic_kitti"
depth_root = "/gpfsscratch/rech/kvd/uyl37fq/Adabin/KITTI/"
preprocess_dir = "/gpfsscratch/rech/kvd/uyl37fq/precompute_data/NYU"
train_dataset = KittiDataset(split="train", 
                             root=kitti_root, 
                             depth_root=depth_root)
val_dataset = KittiDataset(split="val", 
                            root=kitti_root, 
                            depth_root=depth_root)

test_dataset = KittiDataset(split="test", 
                            root=kitti_root, 
                            depth_root=depth_root)

def run(dataset, typ, scale=4):
    TSDF_root = "/gpfsscratch/rech/kvd/uyl37fq/sketch_dataset/TSDF_pred_depth_adabin/kitti"

    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        T_velo_2_cam = batch['T_velo_2_cam']
        P = batch['P']
        name = batch['name']
        sequence = batch['sequence']
        depth_im = batch['depth'].squeeze()[:370, :1220]
        color_im = batch['img'].permute(1, 2, 0).numpy() # Not important
        
        scene_size = (51.2, 51.2, 6.4)
        save_dir = os.path.join(TSDF_root, sequence)
        os.makedirs(save_dir, exist_ok=True)

        vox_origin = np.array([0, -25.6, -2])

#        depth_path = os.path.join(depth_root, name + "_color.png") 
#        depth_im = cv2.imread(depth_path, -1).astype(float)
#        depth_im /= 8000.

#        rgb_path = os.path.join(rgb_root, name + "_color.jpg")
#        color_image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array([scene_size[0], scene_size[1], scene_size[2]])

        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.2 * scale)
        tsdf_vol.integrate(color_im, depth_im, P[0:3, 0:3], np.linalg.inv(T_velo_2_cam), obs_weight=1.)
        
        tsdf_grid, _ = tsdf_vol.get_volume()
        print(tsdf_grid.shape)
#        tsdf_grid = np.moveaxis(tsdf_grid, [0, 1, 2], [0, 2, 1])
        print(np.min(tsdf_grid), np.max(tsdf_grid))
#        tsdf_grid[tsdf_grid > 1.0] = 1.0
#        tsdf_grid[tsdf_grid < -1.0] = -1.0
#        tsdf_path = os.path.join(save_dir, name[3:] + "_1_{}.npy".format(scale))
        tsdf_path = os.path.join(save_dir, name + "_1_{}.npy".format(scale))
        np.save(tsdf_path, tsdf_grid)
        print("Saved to", tsdf_path)


#run(train_dataset, "train", scale=1)
run(test_dataset, "test", scale=1)
#run(train_dataset, "train", scale=4)
run(test_dataset, "test", scale=4)

                  

